// EAGLE Speculative Decoding with KV Cache SSD Offloading
// Based on speculative-eagle.cpp with m-slot circular buffer offloading for target model KV cache
//
// Usage: --kv-offload-slots <m> (default 16)
//        --kv-offload-dir <dir>  (default /tmp/kv_offload)
//        --tree-budget <7|25>    (default 7: small tree, 25: wide tree)
//
// The target model's KV cache uses only m physical slots (m << n_layers).
// During the draft phase, layers 0..m-2 are prefetched from SSD.
// The eval_callback handles layer-level save/prefetch/load orchestration.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-kv-cache-offloaded.h"
#include "../src/llama-cparams.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

#define n_depth 5
#define expand_k 2
#define rerank_k 10

struct callback_data {
    std::vector<float> data;
};

int64_t start_time;

static bool cb_get_hidden(struct ggml_tensor * tensor, bool ask, void * user_data) {
    if (ask) {
        static const char * result_norm_name = "result_norm";
        const bool is_result_norm = strcmp(tensor->name, result_norm_name) == 0;
        start_time = ggml_time_us();
        return is_result_norm;
    }

    int64_t end_time = ggml_time_us();
    int64_t latency = end_time - start_time;
    LOG_DBG("[[Latency for tensor]] '%s' (%s): %lld us\n", tensor->name, ggml_op_name(tensor->op), latency);
    auto * cb_data = (struct callback_data *) user_data;
    auto n_bytes = ggml_nbytes(tensor);
    cb_data->data.resize(n_bytes / sizeof(float));
    ggml_backend_tensor_get(tensor, cb_data->data.data(), 0, n_bytes);
    return true;
}

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};

int main(int argc, char ** argv) {
    common_params params;

    // -------------------------------------------------------------------------
    // Parse KV offloading args before common_params_parse
    // -------------------------------------------------------------------------
    int    n_kv_offload_slots = 16;
    std::string kv_offload_dir = "/tmp/kv_offload";
    int    tree_budget = 7;  // 7 (ver1, small tree) or 25 (ver2, wide tree)

    // Simple pre-scan for our custom args (removed before passing to common_params_parse)
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kv-offload-slots") == 0 && i + 1 < argc) {
            n_kv_offload_slots = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--kv-offload-dir") == 0 && i + 1 < argc) {
            kv_offload_dir = argv[++i];
        } else if (strcmp(argv[i], "--tree-budget") == 0 && i + 1 < argc) {
            tree_budget = std::atoi(argv[++i]);
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int) filtered_argv.size();

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    if (n_kv_offload_slots < 2) {
        LOG_ERR("%s: --kv-offload-slots must be >= 2\n", __func__);
        return 1;
    }

    if (tree_budget != 7 && tree_budget != 25) {
        LOG_ERR("%s: --tree-budget must be 7 or 25\n", __func__);
        return 1;
    }

    fprintf(stderr, "KV offloading: %d pool slots, dir: %s, tree-budget: %d\n",
            n_kv_offload_slots, kv_offload_dir.c_str(), tree_budget);

    const int n_seq_dft = params.n_parallel;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    llama_backend_init();
    llama_numa_init(params.numa);

    callback_data cb_data;
    // Save cache_type_k/v before params is overwritten for draft model loading
    const ggml_type tgt_type_k = params.cache_type_k;
    const ggml_type tgt_type_v = params.cache_type_v;

    // For prefill, cb_get_hidden is used normally (no offloading callback yet)
    params.cb_eval = cb_get_hidden;
    params.cb_eval_user_data = &cb_data;

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    // =========================================================================
    // Install KV cache offloader for target model
    // =========================================================================
    static kv_offload_cb_data offload_cb_data; // must outlive ctx_tgt
    {
        const llama_hparams & hparams_tgt = model_tgt->hparams;
        const llama_cparams & cp          = ctx_tgt->get_cparams();

        // Capture context values before constructor to avoid argument evaluation order issues
        const uint32_t tgt_kv_size  = ctx_tgt->n_ctx_seq();
        const uint32_t tgt_seq_max  = ctx_tgt->n_seq_max();

        // Free the original full KV cache first to reclaim GPU memory
        // before allocating the smaller offloaded KV cache (m pool slots)
        ctx_tgt->set_memory(nullptr);

        auto offloader_ptr = std::unique_ptr<llama_kv_cache_offloaded>(
            new llama_kv_cache_offloaded(
                *model_tgt,
                tgt_type_k,
                tgt_type_v,
                !cp.flash_attn,              // v_trans
                true,                        // offload: KV on GPU for fast attention compute
                cp.kv_unified,               // unified
                tgt_kv_size,                 // kv_size
                tgt_seq_max,                 // n_seq_max
                1,                           // n_pad
                hparams_tgt.n_swa,           // n_swa
                hparams_tgt.swa_type,        // swa_type
                (uint32_t) n_kv_offload_slots,
                kv_offload_dir));

        auto * offloader_raw = offloader_ptr.get();

        // Compose: eval_callback handles kqv_out-{il}, forwards result_norm to cb_get_hidden
        offloader_raw->set_user_callback(cb_get_hidden, &cb_data);

        // Replace context's memory module (context takes ownership)
        // graph_reserve will be called automatically by decode() via memory_update()
        ctx_tgt->set_memory(std::move(offloader_ptr));

        // Switch cb_eval to the offloader's composed callback
        offload_cb_data.offloader = offloader_raw;
        offload_cb_data.user_data = &cb_data;
        ctx_tgt->set_eval_callback(llama_kv_cache_offloaded::eval_callback, &offload_cb_data);

        fprintf(stderr, "KV offloader installed: %d pool slots / %d model layers\n",
                offloader_raw->get_pool_slots(), offloader_raw->get_total_layers());
    }

    // Convenience pointer to the installed offloader
    auto * offloader = static_cast<llama_kv_cache_offloaded *>(ctx_tgt->get_memory());

    // load the draft model
    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }
    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    // draft model doesn't need the offloading callback
    params.cb_eval = cb_get_hidden;
    params.cb_eval_user_data = &cb_data;
    common_init_result llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    if (!model_dft || !ctx_dft) {
        LOG_ERR("%s: failed to load draft model\n", __func__);
        return 1;
    }

    // =========================================================================
    // LM HEAD SHARING
    // =========================================================================
    {
        struct ggml_tensor * tgt_output = llama_get_model(ctx_tgt)->output;
        if (!tgt_output) {
            LOG_ERR("Target model output tensor is NULL - cannot perform LM Head Sharing\n");
            return 1;
        }
        const_cast<struct llama_model *>(llama_get_model(ctx_dft))->output = tgt_output;
        auto * mem_dft = llama_get_memory(ctx_dft);
        llama_memory_clear(mem_dft, false);

        if (llama_get_model(ctx_tgt)->output != llama_get_model(ctx_dft)->output) {
            LOG_ERR("LM HEAD SHARING FAILED\n");
            return 1;
        }
        fprintf(stderr, "LM head sharing: OK\n");
    }

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    const bool vocab_type_dft = llama_vocab_type(vocab_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model\n", __func__);
        return 1;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt)         != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt)         != llama_vocab_eos(vocab_dft)) {
        LOG_ERR("%s: draft model special tokens must match target model\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = std::abs(n_vocab_tgt - n_vocab_dft);
        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: target vocab size %d does not match draft vocab size %d\n",
                    __func__, n_vocab_tgt, n_vocab_dft);
            return 1;
        }
        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            if (std::strcmp(llama_vocab_get_text(vocab_tgt, i), llama_vocab_get_text(vocab_dft, i)) != 0) {
                LOG_ERR("%s: token %d content differs\n", __func__, i);
                return 1;
            }
        }
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();
    const auto t_enc_start = ggml_time_us();

    // =========================================================================
    // Prefill
    // =========================================================================
    offloader->set_prefill(true);

    llama_batch temp_batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    int temp_n_past = 0;
    for (int i = 0; i < (int) inp.size() - 1; i++) {
        common_batch_add(temp_batch_tgt, inp[i], temp_n_past++, { 0 }, true);
    }

    const auto t_prefill_start = ggml_time_us();
    llama_decode(ctx_tgt, temp_batch_tgt);
    const auto t_prefill_end = ggml_time_us();
    ctx_tgt->synchronize();
    std::vector<float> sliced_data(cb_data.data.begin(), cb_data.data.end());

    LOG("\nbatch_tgt.n_tokens: %d, prefill latency: %.3f s\n",
        temp_batch_tgt.n_tokens, (t_prefill_end - t_prefill_start) / 1e6f);

    // After the first prefill decode, pool slots contain stale data from the
    // last layer that mapped to each slot.  Switch to decode mode so the
    // eval_callback restores each layer's data from SSD before it is reused.
    offloader->set_prefill(false);
    offloader->prepare_target_pass();      // queue LOADs for layers 0..m-1
    offloader->wait_and_load_layer(0);     // ensure layer 0 is ready

    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1));
    std::vector<float> backup_data(cb_data.data.begin(), cb_data.data.end());

    llama_decode_eagle(ctx_dft, llama_batch_get_one(inp.data() + 1, n_input - 1), sliced_data.data());

    const auto t_enc_end = ggml_time_us();

    int n_draft = params.speculative.n_max;
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size() - 1;

    bool has_eos = false;

    std::vector<seq_draft> drafts(n_seq_dft);

    std::vector<int>   acceptance_lengths;
    std::vector<float> confidence_scores;
    std::vector<int>   decoding_latencies;
    std::vector<int>   verification_latencies;
    std::vector<float> T_d;
    std::vector<std::vector<int>> accept_counts(n_seq_dft, std::vector<int>(n_depth, 0));

    int rows = n_seq_dft;
    int cols = n_depth;
    std::vector<std::vector<float>> scores(rows, std::vector<float>(cols, 0.0f));
    std::vector<float> column_scores(n_seq_dft, 0.0f);

    int cur_depth = 0;
    int third_depth[4] = { 0, 1, 0, 0 };
    if (tree_budget == 25) {
        third_depth[2] = 4; third_depth[3] = 5;
    }

    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    auto verification_start = ggml_time_us();

    while (true) {
        std::set<int> active_seqs = {};

        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }
            active_seqs.insert(s);
            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, drafts[s].tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        std::vector<float> temp2;
        std::vector<llama_token> recompute;

        // Verification loop
        while (true) {
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);
                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data(), drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        if (r <= p_tgt / p_dft) {
                            s_keep    = s;
                            accept    = true;
                            token_id  = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);
                            break;
                        } else {
                            drafts[s].active = false;
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size,
                                      [](const llama_token_data & a, const llama_token_data & b) { return a.id < b.id; });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size,
                                      [](const llama_token_data & a, const llama_token_data & b) { return a.id < b.id; });
                            float sum_probs = 0.0f;
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - (i < dist_dft.size ? dist_dft.data[i].p : 0.0f));
                                sum_probs += dist_tgt.data[i].p;
                            }
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size,
                                      [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
                        }
                        active_seqs.erase(s);
                        for (int j = 0; j < n_seq_dft; j++) {
                            if (j == s) continue;
                            if (drafts[j].active && drafts[j].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                drafts[j].active = drafts[j].active && accept;
                                if (!drafts[j].active) active_seqs.erase(s);
                            }
                        }
                    }

                    if (!accept) {
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) probs[i] = dist_tgt.data[i].p;
                        std::discrete_distribution<> dist(probs.begin(), probs.end());
                        const int idx = dist(rng);
                        token_id  = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else {
                    // greedy verification
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);
                    common_sampler_accept(smpl, token_id, true);
                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    temp2.insert(temp2.end(),
                                 backup_data.begin() + (4096 * (drafts[s_keep].i_batch_tgt[i_dft])),
                                 backup_data.begin() + (4096 * (drafts[s_keep].i_batch_tgt[i_dft] + 1)));
                    recompute.push_back(token_id);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) continue;
                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            accept_counts[s][i_dft]++;
                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    LOG("%s", token_str.c_str());
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }

        // Notify offloader of accepted token count (for stats)
        offloader->on_verify_complete((uint32_t) i_dft);

        const auto verification_end = ggml_time_us();
        verification_latencies.push_back((int)((verification_end - verification_start) / 1000));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) scores[i][j] = 0.0f;
        }

        acceptance_lengths.push_back(i_dft + 1);

        backup_data = temp2;
        std::vector<float> temp3(backup_data.end() - 4096, backup_data.end());
        int recompute_point = n_past_dft - i_dft;

        /////////////////////////////////////////Drafting Start///////////////////////////////////////

        const auto drafting_start = ggml_time_us();

        // -----------------------------------------------------------------
        // KV offload: start prefetching layers 0..m-2 during draft phase
        // -----------------------------------------------------------------
        offloader->reset_layer_tracking();
        offloader->prepare_target_pass();

        const auto recompute_start = ggml_time_us();
        {
            llama_memory_seq_keep(mem_dft, s_keep);
            llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
            llama_memory_seq_keep(mem_dft, 0);

            llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
            llama_memory_seq_keep(mem_tgt, s_keep);
            llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
            llama_memory_seq_keep(mem_tgt, 0);

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            llama_memory_seq_rm(mem_dft, 0, recompute_point, -1);

            if (i_dft > 0) {
                std::vector<float> temp4(backup_data.begin(), backup_data.end() - 4096);
                common_batch_clear(batch_dft);
                for (int i = 0; i < (int) recompute.size() - 1; i++) {
                    common_batch_add(batch_dft, recompute[i], recompute_point + i, { 0 }, false);
                }
                llama_decode_eagle(ctx_dft, batch_dft, temp4.data());
            }

            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, token_id, n_past_dft, {0}, true);
            llama_decode_eagle(ctx_dft, batch_dft, temp3.data());
            ++n_past_dft;
        }
        const auto recompute_end = ggml_time_us();
        LOG_DBG("recompute took %.3f s\n", (recompute_end - recompute_start) / 1e6f);

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        /////////////////////////////////////////Tree Decoding///////////////////////////////////////

        (void) ggml_time_us(); // tree_decoding_start (timing removed)

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            if (batch_tgt.n_tokens >= n_draft) break;
            if (i >= 5)                         break;

            if (tree_budget == 25) {
                // Budget-25 skip logic
                if (cur_depth < 2) {
                    for (int s = 0; s < n_seq_dft; ++s) drafts[s].skip = false;
                } else if (cur_depth == 2) {
                    for (int s = 0; s < n_seq_dft; ++s) {
                        int in = 0;
                        for (int ii = 0; ii < 4; ii++) if (s == third_depth[ii]) in = 1;
                        drafts[s].skip = (in == 0);
                    }
                } else if (cur_depth == 3 || cur_depth == 4) {
                    for (int s = 0; s < n_seq_dft; ++s) {
                        drafts[s].skip = (s != 0);
                    }
                } else {
                    for (int s = 0; s < n_seq_dft; ++s) drafts[s].skip = false;
                }
            } else {
                // Budget-7 skip logic
                if (cur_depth < 2) {
                    for (int s = 0; s < n_seq_dft; ++s) drafts[s].skip = false;
                } else if (cur_depth == 2) {
                    for (int s = 0; s < n_seq_dft; ++s) {
                        int in = 0;
                        for (int ii = 0; ii < 4; ii++) if (s == third_depth[ii]) in = 1;
                        drafts[s].skip = (in == 0);
                    }
                } else if (cur_depth == 3 || cur_depth == 4) {
                    for (int s = 0; s < n_seq_dft; ++s) {
                        int in = 0;
                        for (int ii = 0; ii < 4; ii++) if (s == third_depth[ii]) in = 1;
                        drafts[s].skip = (in == 0);
                    }
                } else {
                    for (int s = 0; s < n_seq_dft; ++s) drafts[s].skip = false;
                }
            }

            std::vector<float> temp;

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) continue;

                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);

                std::vector<int> sa(1, s);
                temp.insert(temp.end(),
                             cb_data.data.begin() + (4096 * s),
                             cb_data.data.begin() + (4096 * (s + 1)));

                float prob = cur_p->data[0].p;
                if (i == 0) {
                    scores.at(s).at(i) = prob;
                    column_scores.at(s) = prob;
                } else {
                    scores.at(s).at(i) = scores.at(s).at(i-1) * prob;
                    column_scores.at(s) = scores.at(s).at(i-1) * prob;
                }

                int f_max = 1;
                if (tree_budget == 25) {
                    // Budget-25: wide tree
                    if (cur_depth == 0) f_max = 4;
                    else if (cur_depth == 1) {
                        if      (s == 0) f_max = 3;
                        else if (s == 1) f_max = 2;
                        else if (s == 2) f_max = 2;
                        else if (s == 3) f_max = 1;
                    } else if (cur_depth == 2) {
                        if      (s == 0) f_max = 3;
                        else if (s == 1) f_max = 1;
                        else if (s == 4) f_max = 2;
                        else if (s == 5) f_max = 2;
                    } else if (cur_depth == 3) {
                        if (s == 0) f_max = 3;
                    } else if (cur_depth == 4) {
                        f_max = 2;
                    } else {
                        f_max = 4;
                    }
                } else {
                    // Budget-7: small tree
                    if (cur_depth == 0)       f_max = 2;
                    else if (cur_depth == 1)  f_max = (s == 0) ? 1 : 0;
                    else if (cur_depth == 2)  f_max = (s == 0) ? 1 : 0;
                    else if (cur_depth == 3)  f_max = (s == 0) ? 1 : 0;
                    else if (cur_depth == 4)  f_max = 1;
                    else                      f_max = 1;
                }

                for (int f = 1; f < f_max; ++f) {
                    if (n_seq_cur < n_seq_dft) {
                        llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);
                        temp.insert(temp.end(),
                                     cb_data.data.begin() + (4096 * s),
                                     cb_data.data.begin() + (4096 * (s + 1)));
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;
                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;
                        if (drafts[n_seq_cur].smpl) common_sampler_free(drafts[n_seq_cur].smpl);
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);
                        sa.push_back(n_seq_cur);
                        n_seq_cur++;
                        float p2 = cur_p->data[f].p;
                        if (i == 0) {
                            scores.at(n_seq_cur-1).at(i) = p2;
                            column_scores.at(n_seq_cur-1) = p2;
                        } else {
                            scores.at(n_seq_cur-1).at(i) = scores.at(s).at(i-1) * p2;
                            column_scores.at(n_seq_cur-1) = scores.at(s).at(i-1) * p2;
                        }
                    } else {
                        break;
                    }
                }

                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;
                    const int ss = sa[is];
                    common_sampler_accept(drafts[ss].smpl, id, true);
                    drafts[ss].tokens.push_back(id);
                    drafts[ss].dists.push_back({cur_p->data, cur_p->data + cur_p->size});
                    drafts[ss].i_batch_tgt.push_back(batch_tgt.n_tokens);
                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { ss }, true);
                    drafts[ss].i_batch_dft = batch_dft.n_tokens;
                    if (tree_budget == 25) {
                        // Budget-25: batch_dft add logic
                        if (cur_depth == 0) {
                            common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 1) {
                            int in = 0;
                            for (int ii = 0; ii < 4; ii++) if (ss == third_depth[ii]) in = 1;
                            if (in == 1) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 2) {
                            if (ss == 0) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 3) {
                            if (ss == 0) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 4) {
                            if (ss == 0) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        }
                    } else {
                        // Budget-7: batch_dft add logic
                        if (cur_depth == 0) {
                            common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 1) {
                            int in = 0;
                            for (int ii = 0; ii < 2; ii++) if (ss == third_depth[ii]) in = 1;
                            if (in == 1) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 2) {
                            int in = 0;
                            for (int ii = 0; ii < 2; ii++) if (ss == third_depth[ii]) in = 1;
                            if (in == 1) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 3) {
                            if (ss == 0) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                            else drafts[ss].drafting = false;
                        } else if (cur_depth == 4) {
                            if (ss == 0) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                            else drafts[ss].drafting = false;
                        }
                    }
                    if (batch_tgt.n_tokens > n_draft) drafts[ss].drafting = false;
                }
            }

            if (i + 1 == n_depth) {
                float sum = 0.0f;
                for (int ii = 0; ii < rows; ii++)
                    for (int j = 0; j < cols; j++)
                        sum += scores[ii][j];
                confidence_scores.push_back(sum);
            }

            if (batch_dft.n_tokens == 0)         break;
            if (batch_tgt.n_tokens > n_draft)    break;

            const auto dft_decode_start = ggml_time_us();
            llama_decode_eagle(ctx_dft, batch_dft, temp.data());
            ctx_dft->synchronize();
            const auto dft_decode_end = ggml_time_us();
            if (batch_dft.n_tokens == 1)
                T_d.push_back((dft_decode_end - dft_decode_start) / 1000.0f);
            ++n_past_cur;
            ++n_drafted;
            cur_depth += 1;
        }
        cur_depth = 0;

        (void) ggml_time_us(); // tree_decoding_end (timing removed)

        const auto drafting_end = ggml_time_us();
        decoding_latencies.push_back((int)((drafting_end - drafting_start) / 1000.0f));

        verification_start = ggml_time_us();

        // =====================================================================
        // Target verification
        // KV offload: wait_and_load_layer(0) for cold start, then eval_callback
        // handles layers 1..L-1 automatically.
        // =====================================================================
        {
            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
            }

            // Ensure layer 0 is loaded into its tensor slot before graph compute starts
            offloader->wait_and_load_layer(0);

            const auto t_dec_s = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            ctx_tgt->synchronize();
            const auto t_dec_e = ggml_time_us();
            LOG_DBG("batch_tgt.n_tokens: %d, target decode: %.3f s\n",
                    batch_tgt.n_tokens, (t_dec_e - t_dec_s) / 1e6f);
            backup_data = cb_data.data;
            ++n_past_tgt;
        }

        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) continue;
            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    // Print stats
    {
        const auto & st = offloader->get_stats();
        fprintf(stderr, "\nKV offload stats:\n");
        fprintf(stderr, "  prefetch hits  : %u\n", st.prefetch_hits);
        fprintf(stderr, "  prefetch stalls: %u\n", st.prefetch_stalls);
        fprintf(stderr, "  saves          : %u\n", st.saves);
        fprintf(stderr, "  loads          : %u\n", st.loads);
    }

    {
        const double prefill_ms  = (t_enc_end - t_enc_start) / 1000.0;
        const double prefill_tps = n_input / (prefill_ms / 1000.0);
        const double decode_ms   = (t_dec_end - t_dec_start) / 1000.0;
        const double decode_tps  = n_predict / (decode_ms / 1000.0);
        const double decode_lat  = n_predict > 0 ? decode_ms / n_predict : 0;
        const int    n_steps     = (int) decoding_latencies.size();
        const double draft_len   = n_steps > 0 ? (double) n_drafted / n_steps : 0;
        const double accept_len  = n_steps > 0
            ? std::accumulate(acceptance_lengths.begin() + 1, acceptance_lengths.end(), 0.0) / n_steps : 0;
        const double avg_draft_lat  = !decoding_latencies.empty()
            ? std::accumulate(decoding_latencies.begin(), decoding_latencies.end(), 0.0) / decoding_latencies.size() : 0;
        const double avg_verify_lat = !verification_latencies.empty()
            ? std::accumulate(verification_latencies.begin(), verification_latencies.end(), 0.0) / verification_latencies.size() : 0;
        const double avg_td = !T_d.empty()
            ? std::accumulate(T_d.begin(), T_d.end(), 0.0) / T_d.size() : 0;

        fprintf(stderr, "\n");
        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "       EAGLE + KV SSD Offloading  Performance Summary\n");
        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "  Pool slots     : %d / %d model layers\n",
                offloader->get_pool_slots(), offloader->get_total_layers());
        fprintf(stderr, "  Prefill        : %5d tokens | %9.2f ms | %8.2f t/s\n", n_input, prefill_ms, prefill_tps);
        fprintf(stderr, "  Decode         : %5d tokens | %9.2f ms | %8.2f t/s\n", n_predict, decode_ms, decode_tps);
        fprintf(stderr, "  Decode latency :              | %9.2f ms/tok\n", decode_lat);
        fprintf(stderr, "------------------------------------------------------------\n");
        fprintf(stderr, "  Draft length          : %.3f\n", draft_len);
        fprintf(stderr, "  Avg accept length     : %.3f\n", accept_len);
        fprintf(stderr, "  Accept ratio          : %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
        fprintf(stderr, "------------------------------------------------------------\n");
        fprintf(stderr, "  Avg draft phase       : %9.3f ms\n", avg_draft_lat);
        fprintf(stderr, "  Avg verification      : %9.3f ms\n", avg_verify_lat);
        fprintf(stderr, "  Avg T_d (1-tok dft)   : %9.3f ms\n", avg_td);
        fprintf(stderr, "============================================================\n");
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);
    llama_batch_free(temp_batch_tgt);

    llama_backend_free();

    return 0;
}
