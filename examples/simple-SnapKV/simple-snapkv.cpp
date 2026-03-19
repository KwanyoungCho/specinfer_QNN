// simple-snapkv: SnapKV selective KV cache compression demo
//
// Usage:
//   llama-simple-snapkv -m model.gguf -p "your long prompt" -n 128 -ngl 99 \
//       --snapkv-budget 256 --snapkv-obs-window 32
//
// Based on llama-cli (uses the common library for model/context init).
// SnapKV is always enabled in this example.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-kv-cache.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m model.gguf -p \"long prompt\" -n 128 -ngl 99 --snapkv-budget 256 --snapkv-obs-window 32\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    // ── 1. Pre-parse SnapKV-specific args (not known to common_params_parse) ──
    uint32_t snapkv_budget     = 256;
    uint32_t snapkv_obs_window = 32;

    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--snapkv-budget") == 0 && i + 1 < argc) {
            snapkv_budget = (uint32_t) std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--snapkv-obs-window") == 0 && i + 1 < argc) {
            snapkv_obs_window = (uint32_t) std::stoi(argv[++i]);
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    int    filtered_argc = (int) filtered_argv.size();
    char ** filtered_argv_ptr = filtered_argv.data();

    // ── 2. Parse standard args via common library ──
    common_params params;
    if (!common_params_parse(filtered_argc, filtered_argv_ptr, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    // force non-conversation mode for simple generation
    params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;

    if (params.prompt.empty()) {
        params.prompt = "Hello my name is";
    }

    common_init();

    LOG_INF("%s: SnapKV enabled — budget=%u, obs_window=%u\n", __func__, snapkv_budget, snapkv_obs_window);

    // ── 3. Init backend ──
    llama_backend_init();
    llama_numa_init(params.numa);

    // ── 4. Load model ──
    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ── 5. Create context with SnapKV enabled ──
    auto cparams = common_context_params_to_llama(params);
    cparams.snapkv            = true;
    cparams.snapkv_budget     = snapkv_budget;
    cparams.snapkv_obs_window = snapkv_obs_window;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx);
    LOG_INF("%s: n_ctx = %d\n", __func__, n_ctx);

    // ── 6. Tokenize prompt ──
    std::vector<llama_token> embd_inp = common_tokenize(ctx, params.prompt, true, true);
    const int n_prompt = (int) embd_inp.size();

    LOG_INF("%s: prompt tokens = %d\n", __func__, n_prompt);
    if (n_prompt >= n_ctx) {
        LOG_ERR("%s: prompt (%d tokens) exceeds context size (%d)\n", __func__, n_prompt, n_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // ── 7. Init sampler ──
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to init sampler\n", __func__);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // ── 8. Print prompt ──
    for (auto id : embd_inp) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    // ── 9. Prefill: evaluate the full prompt ──
    const int n_batch = llama_n_batch(ctx);
    int n_past = 0;

    for (int i = 0; i < n_prompt; i += n_batch) {
        int n_eval = std::min(n_batch, n_prompt - i);
        llama_batch batch = llama_batch_get_one(&embd_inp[i], n_eval);

        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: failed to decode batch at pos %d\n", __func__, i);
            common_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        n_past += n_eval;
    }

    LOG_INF("\n%s: prefill done (%d tokens)\n", __func__, n_past);

    // ── 9b. SnapKV: cell metadata is already trimmed by llama-context.cpp (seq_rm) ──
    // Just reset n_past to budget so generation tokens are placed right after packed entries.
    {
        n_past = (int) snapkv_budget;
        LOG_INF("%s: n_past reset to budget=%u for generation\n", __func__, snapkv_budget);
    }

    // ── 10. Generation loop ──
    const int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int n_decode = 0;

    const auto t_gen_start = ggml_time_us();

    for (int i = 0; i < n_predict; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);

        if (llama_vocab_is_eog(vocab, id)) {
            LOG("\n");
            break;
        }

        LOG("%s", common_token_to_piece(ctx, id).c_str());
        fflush(stdout); // DEBUG_SNAPKV

        llama_batch batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("\n%s: failed to decode at step %d\n", __func__, i);
            break;
        }

        n_past++;
        n_decode++;
    }

    const auto t_gen_end = ggml_time_us();
    const double t_gen_s = (t_gen_end - t_gen_start) / 1000000.0;

    LOG("\n\n");
    LOG_INF("%s: generated %d tokens in %.2f s (%.2f t/s)\n", __func__, n_decode, t_gen_s,
            n_decode > 0 ? n_decode / t_gen_s : 0.0);

    // ── 11. Print perf stats ──
    LOG_INF("\n");
    common_perf_print(ctx, smpl);

    // ── 12. Cleanup ──
    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
