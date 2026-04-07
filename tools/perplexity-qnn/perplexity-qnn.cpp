#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "QNN/llm_decode_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

struct output_vocab_lookup {
    int32_t n_vocab = 0;
    int32_t n_vocab_full = 0;
    bool identity = true;
    std::vector<int32_t> token_to_output;
    std::vector<int32_t> active_output_indices;

    int32_t output_index(const llama_token tok) const {
        if (tok < 0 || tok >= n_vocab_full) {
            return -1;
        }

        if (identity) {
            return tok < n_vocab ? tok : -1;
        }

        return token_to_output[tok];
    }

    size_t active_size() const {
        return identity ? (size_t) n_vocab : active_output_indices.size();
    }
};

static output_vocab_lookup build_output_vocab_lookup(const llama_model * model, const llama_vocab * vocab) {
    output_vocab_lookup lookup;
    lookup.n_vocab = (int32_t) llama_model_n_vocab_out(model);
    lookup.n_vocab_full = llama_vocab_n_tokens(vocab);
    lookup.identity = lookup.n_vocab == lookup.n_vocab_full;

    if (lookup.identity) {
        for (int32_t i = 0; i < lookup.n_vocab; ++i) {
            if (llama_model_output_token_id(model, i) != i) {
                lookup.identity = false;
                break;
            }
        }
    }

    if (lookup.identity) {
        return lookup;
    }

    lookup.token_to_output.assign(lookup.n_vocab_full, -1);
    lookup.active_output_indices.reserve(lookup.n_vocab);

    for (int32_t i = 0; i < lookup.n_vocab; ++i) {
        const llama_token token_id = llama_model_output_token_id(model, i);
        if (token_id < 0 || token_id >= lookup.n_vocab_full) {
            continue;
        }
        if (lookup.token_to_output[token_id] != -1) {
            continue;
        }

        lookup.token_to_output[token_id] = i;
        lookup.active_output_indices.push_back(i);
    }

    return lookup;
}

static void log_output_vocab_lookup(const char * func, const output_vocab_lookup & lookup) {
    if (lookup.identity) {
        return;
    }

    LOG_WRN("%s: output vocab remap enabled (%zu active outputs, %d logits, tokenizer vocab %d)\n",
            func, lookup.active_size(), lookup.n_vocab, lookup.n_vocab_full);
    if (lookup.active_size() < (size_t) lookup.n_vocab_full) {
        LOG_WRN("%s: tokens outside the trimmed vocab_map receive zero probability and can produce infinite loss\n", func);
    }
}

static void compute_logsumexp(const output_vocab_lookup & lookup, const float * logits, float & max_logit, double & sum_exp) {
    if (lookup.identity) {
        max_logit = logits[0];
        for (int i = 1; i < lookup.n_vocab; ++i) {
            max_logit = std::max(max_logit, logits[i]);
        }
        sum_exp = 0.0;
        for (int i = 0; i < lookup.n_vocab; ++i) {
            sum_exp += expf(logits[i] - max_logit);
        }
        return;
    }

    GGML_ASSERT(!lookup.active_output_indices.empty());
    max_logit = logits[lookup.active_output_indices[0]];
    for (size_t i = 1; i < lookup.active_output_indices.size(); ++i) {
        max_logit = std::max(max_logit, logits[lookup.active_output_indices[i]]);
    }
    sum_exp = 0.0;
    for (const int32_t idx : lookup.active_output_indices) {
        sum_exp += expf(logits[idx] - max_logit);
    }
}

static results_log_softmax log_softmax(const output_vocab_lookup & lookup, const float * logits, llama_token tok) {
    float max_logit = -INFINITY;
    double sum_exp = 0.0;
    compute_logsumexp(lookup, logits, max_logit, sum_exp);

    const int32_t tok_idx = lookup.output_index(tok);
    if (tok_idx < 0) {
        return {-INFINITY, -INFINITY, 0.0f};
    }

    const float tok_logit = logits[tok_idx];
    return {tok_logit - max_logit - log(sum_exp), tok_logit, expf(tok_logit - max_logit) / (float) sum_exp};
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s --qnn --multi-context --ctx-dir ctx_out --tokenizer ctx_out/tokenizer.gguf "
           "--params ctx_out/params.json --log-level 1 -f wiki.test.raw -c 512\n\n", argv[0]);
}

static double perplexity_qnn(
        llama_context * ctx,
        const llama_model * model,
        const llama_vocab * vocab,
        llama_qnn::LLMDecodeRunner & runner,
        std::vector<llama_token> & tokens,
        const int n_ctx,
        const int n_chunks_requested) {

    const output_vocab_lookup lookup = build_output_vocab_lookup(model, vocab);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    log_output_vocab_lookup(__func__, lookup);

    if ((int)tokens.size() < 2 * n_ctx) {
        LOG_ERR("%s: need at least %d tokens to evaluate perplexity with n_ctx=%d, got %zu\n",
                __func__, 2 * n_ctx, n_ctx, tokens.size());
        return -1.0;
    }

    const int n_chunk_max = (int)tokens.size() / n_ctx;
    const int n_chunk = n_chunks_requested < 0 ? n_chunk_max : std::min(n_chunks_requested, n_chunk_max);
    const int first = n_ctx / 2;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    LOG_INF("%s: calculating perplexity over %d chunks, n_ctx=%d\n", __func__, n_chunk, n_ctx);

    for (int i = 0; i < n_chunk; ++i) {
        const int start = i * n_ctx;

        const auto t_start = std::chrono::high_resolution_clock::now();

        runner.reset();
        ctx->final_hiddens.clear();

        // BOS token handling: replace first token of chunk with BOS (same as original perplexity)
        const auto token_org = tokens[start];
        if (add_bos) {
            tokens[start] = llama_vocab_bos(vocab);
        }

        // --- Prefill: tokens[start .. start+first-1] ---
        std::vector<llama_token> prefill_tokens(tokens.begin() + start,
                                                 tokens.begin() + start + first);

        // restore original token
        tokens[start] = token_org;

        llama_batch prefill_batch = llama_batch_get_one(prefill_tokens.data(), (int)prefill_tokens.size());

        if (runner.qnn_decode(ctx, prefill_batch)) {
            LOG_ERR("%s: QNN prefill failed for chunk %d: %s\n", __func__, i, runner.get_error().c_str());
            return -1.0;
        }

        // --- Decode: tokens[start+first .. start+n_ctx-1] one by one, collecting logits ---
        for (int j = first; j < n_ctx - 1; ++j) {
            llama_token tok = tokens[start + j];
            llama_batch decode_batch = llama_batch_get_one(&tok, 1);

            if (runner.qnn_decode(ctx, decode_batch)) {
                LOG_ERR("%s: QNN decode failed at chunk %d pos %d: %s\n",
                        __func__, i, j, runner.get_error().c_str());
                return -1.0;
            }

            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) {
                LOG_ERR("%s: failed to get logits at chunk %d pos %d\n", __func__, i, j);
                return -1.0;
            }

            const int next_tok = tokens[start + j + 1];
            const results_log_softmax results = log_softmax(lookup, logits, next_tok);
            const double v = -results.log_softmax;

            nll  += v;
            nll2 += v * v;
            ++count;
        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        LOG("[%d]%.4lf,", i + 1, std::exp(nll / count));
    }
    LOG("\n");

    nll2 /= count;
    nll  /= count;
    const double ppl = exp(nll);
    nll2 -= nll * nll;
    if (nll2 > 0) {
        nll2 = sqrt(nll2 / (count - 1));
        LOG_INF("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2 * ppl);
    } else {
        LOG_ERR("Unexpected negative standard deviation of log(prob)\n");
    }

    return ppl;
}

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx = 512;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        print_usage(argc, argv);
        return 1;
    }
    common_init();

    const int32_t n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        LOG_ERR("%s: perplexity tool requires '--ctx-size' > 0\n", __func__);
        return 1;
    }

    if (!params.use_qnn) {
        LOG_ERR("%s: this tool requires --qnn flag\n", __func__);
        return 1;
    }

    // --- QNN configuration ---
    llama_qnn::LLMDecodeConfig qnn_config;
    qnn_config.ctx_dir           = params.qnn_ctx_dir;
    qnn_config.backend_so        = params.qnn_backend_so;
    qnn_config.system_so         = params.qnn_system_so;
    qnn_config.tokenizer_path    = params.qnn_tokenizer_path;
    qnn_config.params_path       = params.qnn_params_path;
    qnn_config.max_gen_tokens    = n_ctx;
    qnn_config.log_level         = params.qnn_log_level;
    qnn_config.use_multi_context = params.qnn_use_multi_context;
    qnn_config.num_shards        = params.qnn_num_shards;

    if (qnn_config.ctx_dir.empty() || qnn_config.tokenizer_path.empty()) {
        LOG_ERR("%s: QNN required paths not specified (--ctx-dir / --tokenizer)\n", __func__);
        return 1;
    }
    if (qnn_config.backend_so.empty() || qnn_config.system_so.empty()) {
        qnn_config.backend_so = "libQnnHtp.so";
        qnn_config.system_so  = "libQnnSystem.so";
    }

    // --- Load tokenizer (vocab only) ---
    llama_model_params tokenizer_param = llama_model_default_params();
    tokenizer_param.vocab_only = true;

    llama_model * model = llama_model_load_from_file(qnn_config.tokenizer_path.c_str(), tokenizer_param);
    if (!model) {
        LOG_ERR("%s: unable to load tokenizer model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // --- Create llama context (for logits buffer) ---
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_ctx;
    ctx_params.n_batch = n_ctx;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("%s: failed to create llama_context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    // --- Initialize QNN runner ---
    const auto t_init_start = ggml_time_us();
    llama_qnn::LLMDecodeRunner runner(qnn_config);
    if (!runner.initialize()) {
        LOG_ERR("%s: failed to initialize QNN runner: %s\n", __func__, runner.get_error().c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    const auto t_init_end = ggml_time_us();
    LOG_INF("[QNN] Initialize time: %.2f ms\n", (t_init_end - t_init_start) / 1000.0);

    // --- Read and tokenize input ---
    if (params.prompt.empty()) {
        LOG_ERR("%s: no input text provided. Use -f <file> to specify the evaluation data.\n", __func__);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    LOG_INF("%s: tokenizing the input ..\n", __func__);
    const auto t_tok_start = std::chrono::high_resolution_clock::now();

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    const auto t_tok_end = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %.2f ms, %zu tokens\n", __func__,
            1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t_tok_end - t_tok_start).count(),
            tokens.size());

    // --- Run perplexity ---
    const double ppl = perplexity_qnn(ctx, model, vocab, runner, tokens, n_ctx, params.n_chunks);

    LOG("\n");
    if (ppl > 0) {
        LOG_INF("Perplexity: %.4lf\n", ppl);
    }

    llama_free(ctx);
    llama_model_free(model);

    return (ppl > 0) ? 0 : 1;
}
