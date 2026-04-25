//EAGLE-2 QNN 구현 코드
//Target model runs on QNN NPU, Draft model runs on CPU/GPU via llama.cpp
//-ym-

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "llm_decode_runner.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5


struct callback_data {
    std::vector<float> data;
};

int64_t start_time;

static bool cb_get_hidden(struct ggml_tensor * tensor, bool ask, void * user_data) {
    if (ask) {
        static const char * result_norm_name = "result_norm";
        const bool is_result_norm = strcmp(tensor->name, result_norm_name) == 0;

        return is_result_norm;
    }

    auto * cb_data = (struct callback_data *) user_data;
    auto n_bytes = ggml_nbytes(tensor);
    size_t prev_size = cb_data->data.size();
    cb_data->data.resize(prev_size + n_bytes / sizeof(float));
    ggml_backend_tensor_get(tensor, cb_data->data.data() + prev_size, 0, n_bytes);

    return true;
}

std::vector<size_t> TopK(const std::vector<float>& data, size_t k) {
    size_t n = data.size();

    if (k > n) {
        k = n;
    }

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), 
        indices.begin() + k,
        indices.end(),
        // 람다 함수를 이용한 비교: data의 값을 기준으로 내림차순 정렬
        [&data](size_t a, size_t b) {
            return data[a] > data[b];
        }
    );

    indices.resize(k);
    
    return indices;
}

struct seq_draft { //각 드래프트 시퀀스(트리의 브랜치)의 상태를 저장하는 구조체 -ym-
    bool active   = false; //verification 단계에서 시퀀스가 활성화되었는지 여부 -ym-
    bool drafting = false; //drafting 단계에서 시퀀스가 활성화되었는지 여부 -ym-
    bool skip     = false; //drafting 단계에서 이 시퀀스를 건너뛸지 여부 -ym-

    int i_batch_dft = 0; //드래프트 모델의 배치에서 이 시퀀스의 마지막 토큰 인덱스 -ym-
    std::vector<int> i_batch_tgt; //타겟 모델의 배치에서 이 시퀀스에 해당하는 토큰들의 인덱스 -ym-

    std::vector<llama_token> tokens; //이 시퀀스가 추측한 토큰들의 목록 -ym-
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};

int main(int argc, char ** argv) {
    // ---- Draft Tree Expansion CLI 인자 파싱 시작 ----
    int n_depth = 5;
    int draft_top_k = 10;
    int expand_k = 10;
    bool rerank = true;
    int rerank_k = 59;
    int draft_target_delay_ms = 0;
    int target_draft_delay_ms = 0;

    std::vector<char *> new_argv;
    new_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n-depth" && i + 1 < argc) {
            n_depth = std::stoi(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            draft_top_k = std::stoi(argv[++i]);
            expand_k = draft_top_k; // expand-k도 top-k와 같은 값을 사용하도록 수정
        } else if (arg == "--expand-k" && i + 1 < argc) {
            expand_k = std::stoi(argv[++i]);
        } else if (arg == "--rerank-k" && i + 1 < argc) {
            rerank_k = std::stoi(argv[++i]);
        } else if (arg == "--handoff-delay-ms" && i + 1 < argc) {
            const int delay_ms = std::max(0, std::stoi(argv[++i]));
            draft_target_delay_ms = delay_ms;
            target_draft_delay_ms = delay_ms;
        } else if (arg == "--draft-target-delay-ms" && i + 1 < argc) {
            draft_target_delay_ms = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--target-draft-delay-ms" && i + 1 < argc) {
            target_draft_delay_ms = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--no-rerank") {
            rerank = false;
        } else if (arg == "--rerank") {
            rerank = true;
        } else if (arg == "--help" || arg == "-h") {
            printf("\nDraft Tree Expansion Options:\n");
            printf("  --n-depth N        Draft tree depth (default: 5)\n");
            printf("  --top-k N          Draft tree Top-K (default: 10)\n");
            printf("  --expand-k N       Draft tree Expand-K (default: 10)\n");
            printf("  --rerank-k N       Token-level Reranking K (default: 59)\n");
            printf("  --handoff-delay-ms N      Sleep N ms on both draft<->target handoffs (default: 0)\n");
            printf("  --draft-target-delay-ms N Sleep N ms after GPU draft before NPU verify (default: 0)\n");
            printf("  --target-draft-delay-ms N Sleep N ms after NPU verify before next GPU draft (default: 0)\n");
            printf("  --no-rerank        Disable token-level reranking\n\n");
            new_argv.push_back(argv[i]); // pass to base parser
        } else {
            new_argv.push_back(argv[i]);
        }
    }
    int new_argc = new_argv.size();
    char ** new_argv_ptr = new_argv.data();
    // ---- CLI 인자 파싱 끝 ----

    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(new_argc, new_argv_ptr, params, LLAMA_EXAMPLE_SPECULATIVE)) {
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

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    // const float p_draft_split = params.speculative.p_split;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    callback_data cb_data; //callback data 구조체 변수 선언 -ym-
    params.cb_eval = cb_get_hidden; //callback function 등록 -ym-
    params.cb_eval_user_data = &cb_data; //callback function의 return 값을 callback data 구조체 변수로 받음 -ym-

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // Initialize QNN runner for target model
    llama_qnn::LLMDecodeConfig qnn_config;
    qnn_config.ctx_dir           = params.qnn_ctx_dir;
    qnn_config.backend_so        = params.qnn_backend_so.empty() ? "libQnnHtp.so" : params.qnn_backend_so;
    qnn_config.system_so         = params.qnn_system_so.empty() ? "libQnnSystem.so" : params.qnn_system_so;
    qnn_config.tokenizer_path    = params.qnn_tokenizer_path;
    qnn_config.params_path       = params.qnn_params_path;
    qnn_config.max_gen_tokens    = params.n_predict > 0 ? params.n_predict : 100;
    qnn_config.log_level         = params.qnn_log_level;
    qnn_config.use_multi_context = params.qnn_use_multi_context;
    qnn_config.num_shards        = params.qnn_num_shards;
    qnn_config.deferred_kv_writeback = params.qnn_deferred_kv_writeback;

    // load target model (vocab_only for QNN)
    llama_model_params tgt_model_param = llama_model_default_params();
    tgt_model_param.vocab_only = true;
    model_tgt = llama_model_load_from_file(qnn_config.tokenizer_path.c_str(), tgt_model_param);
    llama_context_params tgt_ctx_param = llama_context_default_params();
    tgt_ctx_param.n_ctx     = 4096;
    tgt_ctx_param.n_batch   = 2048;
    tgt_ctx_param.n_seq_max = params.n_parallel;
    ctx_tgt = llama_init_from_model(model_tgt, tgt_ctx_param);

    if (qnn_config.ctx_dir.empty()) {
        LOG_ERR("%s: --qnn-ctx-dir is required for QNN target model\n", __func__);
        return 1;
    }

    llama_qnn::LLMDecodeRunner qnn_runner(qnn_config);
    if (!qnn_runner.initialize()) {
        LOG_ERR("%s: failed to initialize QNN runner: %s\n", __func__, qnn_runner.get_error().c_str());
        return 1;
    }
    LOG_INF("[QNN] Target model runner initialized successfully\n");
    if (draft_target_delay_ms > 0 || target_draft_delay_ms > 0) {
        LOG_INF("[DIAG] Inter-device handoff delay enabled: draft->target=%d ms, target->draft=%d ms\n",
                draft_target_delay_ms, target_draft_delay_ms);
    }

    // load the draft model
    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    //params.cb_eval = cb_get_latency;
    common_init_result llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    // LM HEAD SHARING not needed - target model runs on QNN NPU

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    // EAGLE vocab trimming: load vocab_map from draft model
    const auto & dft_vocab_map = model_dft->vocab_map;
    const bool has_vocab_trim = !dft_vocab_map.empty();
    if (has_vocab_trim) {
        LOG_INF("[EAGLE] Vocab trimming active: %zu entries in vocab_map (output_vocab_size=%u)\n",
                dft_vocab_map.size(), model_dft->hparams.n_vocab_output);
    }

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    const bool vocab_type_dft = llama_vocab_type(vocab_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        // Skip vocab size difference check when vocab trimming is active
        if (!has_vocab_trim && vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        // Skip per-token text check when vocab trimming is active (tokenizer is shared, only lm_head is trimmed)
        if (!has_vocab_trim) {
            for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
                const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
                const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
                if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                    LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                    LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                            common_token_to_piece(ctx_tgt, i).c_str(),
                            common_token_to_piece(ctx_dft, i).c_str());
                    return 1;
                }
            }
        }
    }

    // mem_tgt not needed - QNN uses internal kv_manager_ for KV cache tracking
    auto * mem_dft = llama_get_memory(ctx_dft);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    // target model sampling context (reuse the llama_context's sampling instance)
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

    // Target model: use QNN prefill (single call for all prompt tokens)
    // KV cache metadata is automatically updated inside qnn_decode
    // const auto t_prefill_start = ggml_time_us();
    {
        llama_batch prefill_batch = llama_batch_get_one(inp.data(), n_input);
        if (qnn_runner.qnn_decode(ctx_tgt, prefill_batch)) {
            LOG_ERR("%s: QNN prefill failed: %s\n", __func__, qnn_runner.get_error().c_str());
            return 1;
        }
    }
    // const auto t_prefill_end = ggml_time_us();

    // Extract hidden states from QNN prefill output (stored in ctx_tgt->final_hiddens)
    const auto& final_hs = ctx_tgt->final_hiddens;
    if (final_hs.empty()) {
        LOG_ERR("%s: QNN prefill did not produce hidden states (final_hiddens is empty)\n", __func__);
        return 1;
    }
    const int hidden_dim = 4096;

    // sliced_data: hidden states for tokens[0..n_input-2] (EAGLE draft uses shifted input)
    std::vector<float> sliced_data(final_hs.begin(), final_hs.begin() + (n_input - 1) * hidden_dim);
    // backup_data: all n_input hidden states from prefill
    // i_batch_tgt[0] = n_input-1 indexes the last prompt token's hidden state
    std::vector<float> backup_data = final_hs;

    // Draft model prefill with EAGLE hidden state sharing
    cb_data.data.clear();
    llama_decode_eagle(ctx_dft, llama_batch_get_one(inp.data() + 1, n_input - 1), sliced_data.data());

    // Reset eagle-decode perf counters after prefill so per-step averages
    // in the Latency Breakdown only reflect the decode loop, not prefill.
    llama_perf_eagle_draft_reset(ctx_dft);

    LOG("\n");LOG("\n");

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // how many tokens to draft each time
    int n_draft = params.speculative.n_max;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    int total_draft_tokens = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size() - 1;

    // used to determine end of generation
    bool has_eos = false;

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    // 각 단계별 수락 길이를 저장하기 위한 벡터
    std::vector<int> acceptance_lengths;
    std::vector<float> confidence_scores;
    std::vector<double> decoding_latencies;
    std::vector<double> verification_latencies;
    std::vector<double> T_d;
    // std::vector<float> tgt_smpl_latencies;
    // std::vector<float> dft_smpl_latencies;
    std::vector<int> temp_i_batch_dft(n_seq_dft, 0);

    int rows = n_seq_dft;
    int cols = n_depth;
    std::vector<std::vector<float>> scores(rows, std::vector<float>(cols, 0.0f));
    std::vector<std::vector<int>> accept_counts(rows, std::vector<int>(cols, 0));

    std::vector<float> column_scores(n_seq_dft, 0.0f);
    std::vector<size_t> topk_indices = { 0, };
    std::vector<size_t> expandk_indices = { 0, };

    // LOG("\nDecoding Starts with: ");

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us(); // 디코딩(생성) 시작 시간 측정

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = n_input - 1;

    auto verification_start = ggml_time_us();

    // Latency breakdown variables (in microseconds)
    int64_t total_draft_wall_us = 0;
    int64_t total_draft_recompute_us = 0;
    int64_t total_draft_setup_us = 0;
    int64_t total_draft_forward_us = 0;
    int64_t total_draft_forward_submit_us = 0; // time from decode_eagle entry until it returns (pre-sync)
    int64_t total_draft_forward_sync_us = 0;   // time spent in ctx->synchronize() right after decode_eagle
    int32_t total_draft_forward_calls = 0;     // number of tree-expansion decode_eagle calls timed here
    int64_t total_draft_recompute_state_reset_us = 0;
    int64_t total_draft_recompute_replay_decode_us = 0;
    int64_t total_draft_recompute_current_decode_us = 0;
    int64_t total_draft_recompute_steps = 0;
    int64_t total_draft_recompute_replay_steps = 0;
    
    // New fine-grained variables for Tree Expansion
    int64_t total_expansion_loop_prep_us = 0;
    int64_t total_expansion_sampling_us = 0;
    int64_t total_expansion_candidate_bookkeeping_us = 0;
    int64_t total_expansion_post_batch_us = 0;
    
    // Splitting Sequence Breakdown Variables
    int64_t total_split_kv_copy_us = 0;
    int64_t total_split_history_update_us = 0;
    int64_t total_split_draft_state_alloc_us = 0;

    int64_t total_expansion_temp_probs_us = 0;
    int64_t total_expansion_topk_us = 0;
    int64_t total_expansion_target_batch_us = 0;

    int64_t total_tree_pruning_us = 0;

    std::vector<int64_t> per_depth_total_us(n_depth, 0);
    std::vector<int64_t> per_depth_body_us(n_depth, 0);
    std::vector<int64_t> per_depth_forward_us(n_depth, 0);
    std::vector<int64_t> per_depth_visit_counts(n_depth, 0);
    std::vector<int64_t> per_depth_forward_calls(n_depth, 0);

    // int64_t total_target_forward_us = 0;
    std::vector<int64_t> target_forward_us;
    int64_t total_target_kv_cache_us = 0;
    int64_t total_verify_logic_us = 0;
    int64_t total_fallback_sampling_us = 0;
    common_sampler_profile_snapshot total_fallback_sampler_profile = {};
    int64_t total_verification_wall_us = 0;
    int64_t total_target_batch_tokens = 0;
    int64_t total_target_batch_seq_refs = 0;
    int64_t total_target_batch_shared_tokens = 0;
    int64_t total_target_tree_attn_edges = 0;
    int64_t total_target_seq_compare_ops = 0;
    int64_t total_target_slot_search_us = 0;
    int64_t total_target_mask_build_us = 0;
    int64_t total_target_shard_prefill_us = 0;
    int64_t total_target_shard_kv_override_us = 0;
    int64_t total_target_shard_input_fill_us = 0;
    int64_t total_target_shard_tensor_build_us = 0;
    int64_t total_target_shard_execute_us = 0;
    int64_t total_target_shard_output_copy_us = 0;
    int64_t total_target_internal_kv_writeback_us = 0;
    int64_t total_target_cell_meta_us = 0;
    int64_t total_target_logits_dequant_us = 0;
    int64_t total_target_hidden_copy_us = 0;
    int64_t total_target_logits_inject_us = 0;
    int32_t max_target_batch_tokens = 0;
    int32_t max_target_batch_seq_refs_per_token = 0;
    int64_t max_target_tree_attn_edges = 0;
    int64_t max_target_seq_compare_ops = 0;
    int64_t total_draft_target_delay_us = 0;
    int64_t total_target_draft_delay_us = 0;
    int num_steps = 0;

    common_sampler_profile_reset();

    while (true) {
        int64_t step_fallback_sampling_us = 0;
        const auto step_verify_logic_start = ggml_time_us();
        std::set<int> active_seqs = {};

        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) { //active 변수의 초기 값은 false, 따라서 첫 prefill 후에는 이 반복문 동작 안함 -ym-
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            // LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        std::vector<float> temp2;
        std::vector<llama_token> recompute;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        // randomly select a sequence to verify from active sequences
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            // if we already accepted a token, we can skip the rest
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        // LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // acquire the token probabilities assigned by the draft and target models
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
                        // LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            // LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            // LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // calculate residual probability
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // sort dist by id
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }

                                sum_probs += dist_tgt.data[i].p;
                            }

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // sort dist_tgt by p desc
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // synchronize active status for sequences with the same drafted token
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        const auto fallback_start = ggml_time_us();
                        // all drafted tokens were rejected
                        // sample from the target model
                        // LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                        step_fallback_sampling_us += (ggml_time_us() - fallback_start);
                    }
                } else {
                    // greedy verification

                    // sample from the target model
                    // LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    const auto fallback_start = ggml_time_us();
                    const auto fallback_profile_start = common_sampler_profile_get();
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);
                    common_sampler_profile_accumulate(
                            total_fallback_sampler_profile,
                            common_sampler_profile_diff(common_sampler_profile_get(), fallback_profile_start));

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);
                    step_fallback_sampling_us += (ggml_time_us() - fallback_start);

                    temp2.insert(temp2.end(), backup_data.begin() + (4096 * (drafts[s_keep].i_batch_tgt[i_dft])), backup_data.begin() + (4096 * (drafts[s_keep].i_batch_tgt[i_dft] + 1)));
                    recompute.push_back(token_id);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            // LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());
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
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }
        
        const auto verification_end = ggml_time_us(); //verification 종료 시간 기록 -ym-
        total_verification_wall_us += (verification_end - verification_start);
        total_verify_logic_us += ((verification_end - step_verify_logic_start) - step_fallback_sampling_us);
        total_fallback_sampling_us += step_fallback_sampling_us;

        double verification_latency = (verification_end - verification_start) / 1000.0; //ms 단위로 변환 -ym-
        verification_latencies.push_back(verification_latency);

        if (target_draft_delay_ms > 0) {
            const auto delay_start = ggml_time_us();
            std::this_thread::sleep_for(std::chrono::milliseconds(target_draft_delay_ms));
            total_target_draft_delay_us += (ggml_time_us() - delay_start);
        }

        for (auto& row : scores) {
            std::fill(row.begin(), row.end(), 0.0f);
        }

        //현재 단계의 수락 길이를 저장
        acceptance_lengths.push_back(i_dft + 1);

        backup_data = temp2;
        std::vector temp3 = std::vector<float>(backup_data.end() - hidden_dim, backup_data.end());
        int recompute_point = n_past_dft - i_dft;

        topk_indices = { 0, };

        /////////////////////////////////////////Drafting Start///////////////////////////////////////

        const auto drafting_start = ggml_time_us(); //tree decoding 시작 시간 기록 -ym-
        // LOG_DBG("Current n_accept: %d, n_drafted: %d, n_predict: %d\n", n_accept, n_drafted, n_predict);

        //////////////////////////////////////////Recompute Logic Start////////////////////////////////////////
        const auto step_recompute_start = ggml_time_us();
        {
            // LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());
            const auto recompute_state_reset_start = ggml_time_us();
            // TODO: simplify
            {
                // LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                llama_memory_seq_keep(mem_dft, s_keep);
                llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_dft, 0);

                // QNN KV cache management for target model
                qnn_runner.kv_seq_rm  (s_keep, n_past_tgt, -1);
                qnn_runner.kv_seq_keep(s_keep);
                qnn_runner.kv_seq_cp  (s_keep, 0, -1, -1);
                qnn_runner.kv_seq_keep(0);
                // Commit KV write-back after metadata cleanup
                if (qnn_runner.is_pending_KV_write()) {
                    qnn_runner.KV_commit();
                }
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            llama_memory_seq_rm(mem_dft, 0, recompute_point, -1);
            const auto recompute_state_reset_end = ggml_time_us();
            total_draft_recompute_state_reset_us += (recompute_state_reset_end - recompute_state_reset_start);

            //recompute logic 추가 -ym-
            if (i_dft > 0) {
                ++total_draft_recompute_replay_steps;
                const auto recompute_replay_decode_start = ggml_time_us();
                std::vector temp4 = std::vector<float>(backup_data.begin(), backup_data.end() - hidden_dim);

                common_batch_clear(batch_dft);
                for (size_t i = 0; i < recompute.size() - 1; i++) {
                    common_batch_add  (batch_dft, recompute[i], recompute_point + i, { 0 }, false);
                }
                // const auto recompute_decode_start = ggml_time_us();
                cb_data.data.clear();
                llama_decode_eagle(ctx_dft, batch_dft, temp4.data());
                // const auto recompute_decode_end = ggml_time_us();
                const auto recompute_replay_decode_end = ggml_time_us();
                total_draft_recompute_replay_decode_us += (recompute_replay_decode_end - recompute_replay_decode_start);
            }

            const auto recompute_current_decode_start = ggml_time_us();
            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, token_id, n_past_dft, {0}, true);

            // const auto recompute_decode_start1 = ggml_time_us();
            cb_data.data.clear();
            llama_decode_eagle(ctx_dft, batch_dft, temp3.data());
            // const auto recompute_decode_end1 = ggml_time_us();
            const auto recompute_current_decode_end = ggml_time_us();
            total_draft_recompute_current_decode_us += (recompute_current_decode_end - recompute_current_decode_start);

            ++n_past_dft;
        }

        //////////////////////////////////////////Recompute Logic End////////////////////////////////////////
        const auto step_recompute_end = ggml_time_us();
        total_draft_recompute_us += (step_recompute_end - step_recompute_start);
        ++total_draft_recompute_steps;

        const auto step_draft_setup_start = ggml_time_us();
        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // Context overflow guard: stop if approaching draft model KV cache limit
        {
            const int n_ctx_dft = llama_n_ctx(ctx_dft);
            if (n_past_dft + n_draft + n_depth >= n_ctx_dft - 2) {
                fprintf(stderr, "\n[WARN] Draft model context nearly full (n_past_dft=%d, n_ctx=%d). Stopping.\n",
                        n_past_dft, n_ctx_dft);
                break;
            }
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

            // [추가] 0번 루트 시퀀스를 제외한 나머지 비활성 시퀀스의 sampler를 즉시 해제합니다.
            if (s > 0 && drafts[s].smpl != nullptr) {
                common_sampler_free(drafts[s].smpl);
                drafts[s].smpl = nullptr;
            }
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        /////////////////////////////////////////Tree Decoding Start///////////////////////////////////////
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        expandk_indices = { 0, };
        const auto step_draft_setup_end = ggml_time_us();
        total_draft_setup_us += (step_draft_setup_end - step_draft_setup_start);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            const auto step_loop_prep_start = ggml_time_us();
            batch_dft.n_tokens = 0;
            for (int i = 0; i < rows; i++) {
                column_scores[i] = 0;
            }

            if (batch_tgt.n_tokens >= n_draft) {
                break;
            }

            const auto depth_total_start = ggml_time_us();
            ++per_depth_visit_counts[i];

            for (int s = 0; s < n_seq_dft; ++s) {
                auto it_last = std::find(topk_indices.begin(), topk_indices.end(), s);
                if (it_last != topk_indices.end()) {
                    drafts[s].skip = false;
                } else {
                    drafts[s].skip = true;
                }
            }

            std::vector<float> temp;
            std::vector<llama_token> ids;
            std::vector<int> ss;
            std::vector<float> temp_probs;
            std::vector<std::vector<llama_token_data>> datas;
            const auto step_loop_prep_end = ggml_time_us();
            total_expansion_loop_prep_us += (step_loop_prep_end - step_loop_prep_start);

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                // LOG_DBG("drafting sequence %d at pos %d\n", s, i);

                ////////////////////////////////////////Sampling Start///////////////////////////////////////

                const auto t_samp_start = ggml_time_us();
                
                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);
                const auto t_samp_end = ggml_time_us();
                total_expansion_sampling_us += (t_samp_end - t_samp_start);

                // EAGLE vocab trimming: remap candidate IDs from trimmed logits index to original token IDs
                if (has_vocab_trim) {
                    for (size_t ci = 0; ci < cur_p->size; ++ci) {
                        const int idx = cur_p->data[ci].id;
                        if (idx >= 0 && idx < (int)dft_vocab_map.size()) {
                            cur_p->data[ci].id = dft_vocab_map[idx];
                        }
                    }
                }

                const auto candidate_bookkeeping_start = ggml_time_us();
                // for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                //     LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                //             k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                // }

                std::vector<int> sa(1, s);

                // temp.insert(temp.end(), cb_data.data.begin() + (hidden_dim * drafts[s].i_batch_dft), cb_data.data.begin() + (hidden_dim * (drafts[s].i_batch_dft + 1)));

                /////////////////////////////////////////Sampling End///////////////////////////////////////

                // Accumulated Probability Table Add 1
                float prob = cur_p->data[0].p;
                // LOG_DBG(" %f \n", prob);
                if (i == 0) {
                    scores.at(s).at(i) = prob;
                    column_scores.at(s) = prob;
                }
                else {
                    scores.at(s).at(i) = scores.at(s).at(i-1) * prob;
                    column_scores.at(s) = scores.at(s).at(i-1) * prob;
                }
                const auto candidate_bookkeeping_end = ggml_time_us();
                total_expansion_candidate_bookkeeping_us += (candidate_bookkeeping_end - candidate_bookkeeping_start);

                ////////////////////////////////////////Split Start///////////////////////////////////////
                for (int f = 1; f < expand_k; ++f) {
                    // LOG_DBG("cur_p->data[f].p = %lf\n", cur_p->data[f].p);
                    // if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                    if (n_seq_cur < n_seq_dft) {
                        // LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        const auto t_split_kv_start = ggml_time_us();
                        llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);
                        const auto t_split_kv_end = ggml_time_us();
                        total_split_kv_copy_us += (t_split_kv_end - t_split_kv_start);

                        const auto t_split_history_start = ggml_time_us();
                        // all previous tokens from this branch are now also part of the new branch
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }
                        const auto t_split_history_end = ggml_time_us();
                        total_split_history_update_us += (t_split_history_end - t_split_history_start);

                        const auto t_split_state_start = ggml_time_us();
                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);
                        sa.push_back(n_seq_cur);
                        n_seq_cur++;

                        // Accumulated Probability Table Add 2
                        float prob = cur_p->data[f].p;
                        if (i == 0) {
                            scores.at(n_seq_cur-1).at(i) = prob;
                            column_scores.at(n_seq_cur-1) = prob;
                        }
                        else {
                            scores.at(n_seq_cur-1).at(i) = scores.at(s).at(i-1) * prob;
                            column_scores.at(n_seq_cur-1) = scores.at(s).at(i-1) * prob;
                        }
                        const auto t_split_state_end = ggml_time_us();
                        total_split_draft_state_alloc_us += (t_split_state_end - t_split_state_start);
                    } else {
                        break;
                    }
                }

                ////////////////////////////////////////Split End///////////////////////////////////////

                ////////////////////////////////////////Add Tokens Start///////////////////////////////////////
                const auto t_temp_prob_start = ggml_time_us();
                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;
                    ids.push_back(id);
                    temp_probs.push_back(cur_p->data[is].p);
                    datas.push_back({cur_p->data, cur_p->data + cur_p->size});

                    const int s = sa[is];
                    ss.push_back(s);
                }

                for (int i = 0; i < n_seq_dft; i++) {
                    temp_i_batch_dft[i] = drafts[i].i_batch_dft;
                }

                const auto t_temp_prob_end = ggml_time_us();
                total_expansion_temp_probs_us += (t_temp_prob_end - t_temp_prob_start);

                ////////////////////////////////////////Add Tokens End///////////////////////////////////////
            }

            const auto topk_start = ggml_time_us();
            expandk_indices = TopK(temp_probs, expand_k);
            topk_indices = TopK(column_scores, draft_top_k);
            const auto topk_end = ggml_time_us();
            total_expansion_topk_us += (topk_end - topk_start);

            const auto target_batch_start = ggml_time_us();
            for (int is = 0; is < (int) ids.size(); ++is) {
                const llama_token id = ids[is];
                const int s = ss[is];
                const float p = temp_probs[is];
                const auto cur_p = &datas[is];

                common_sampler_accept(drafts[s].smpl, id, true);
                drafts[s].tokens.push_back(id);

                // add unique drafted tokens to the target batch
                drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);
                common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);
                // LOG_DBG("batch_tgt.n_tokens: %d\n", batch_tgt.n_tokens);

                if (batch_tgt.n_tokens >= n_draft)
                    break;

                // add the token to the batch for batched decoding with the draft model
                if (batch_dft.n_tokens >= draft_top_k)
                    drafts[s].i_batch_dft = draft_top_k - 1;
                else
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                if (topk_indices.size() == 1) {
                    common_batch_add(batch_dft, id, n_past_cur, {s}, true);
                    temp.insert(temp.end(), cb_data.data.begin() + (hidden_dim * temp_i_batch_dft[s]), cb_data.data.begin() + (hidden_dim * (temp_i_batch_dft[s] + 1)));
                }
                else {
                    auto it_last = std::find(topk_indices.begin(), topk_indices.end(), s);
                    if (it_last != topk_indices.end()) {
                        common_batch_add(batch_dft, id, n_past_cur, {s}, true);
                        temp.insert(temp.end(), cb_data.data.begin() + (hidden_dim * temp_i_batch_dft[s]), cb_data.data.begin() + (hidden_dim * (temp_i_batch_dft[s] + 1)));
                    }
                }

                if (batch_tgt.n_tokens > n_draft) {
                    drafts[s].drafting = false;
                }
            }
            const auto target_batch_end = ggml_time_us();
            total_expansion_target_batch_us += (target_batch_end - target_batch_start);

            const auto post_batch_start = ggml_time_us();
            for (int i = 0; i < n_seq_dft; i++) {
                    temp_i_batch_dft[i] = drafts[i].i_batch_dft;
            }

            if (i + 1 == n_depth) {
                float sum = 0.0f;
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        sum += scores[i][j];
                    }
                }
                confidence_scores.push_back(sum);
            }

            const bool stop_due_depth = (i + 1 >= n_depth);
            const bool stop_due_no_draft_batch = (batch_dft.n_tokens == 0);
            const bool stop_due_target_overflow = (batch_tgt.n_tokens > n_draft);
            const auto post_batch_end = ggml_time_us();
            total_expansion_post_batch_us += (post_batch_end - post_batch_start);
            const int64_t depth_body_us = post_batch_end - depth_total_start;

            if (stop_due_depth || stop_due_no_draft_batch || stop_due_target_overflow) {
                per_depth_body_us[i] += depth_body_us;
                per_depth_total_us[i] += depth_body_us;
                break;
            }

            // evaluate the drafted tokens on the draft model
            const auto dft_model_decode_start = ggml_time_us();
            cb_data.data.clear();
            llama_decode_eagle(ctx_dft, batch_dft, temp.data());
            const auto dft_model_decode_submit_end = ggml_time_us();
            ctx_dft->synchronize();
            const auto dft_model_decode_end = ggml_time_us();
            total_draft_forward_us += (dft_model_decode_end - dft_model_decode_start);
            total_draft_forward_submit_us += (dft_model_decode_submit_end - dft_model_decode_start);
            total_draft_forward_sync_us += (dft_model_decode_end - dft_model_decode_submit_end);
            ++total_draft_forward_calls;
            per_depth_body_us[i] += depth_body_us;
            per_depth_forward_us[i] += (dft_model_decode_end - dft_model_decode_start);
            per_depth_total_us[i] += (dft_model_decode_end - depth_total_start);
            ++per_depth_forward_calls[i];
            T_d.push_back((dft_model_decode_end - dft_model_decode_start) / 1000.0f);
            ++n_past_cur;
            ++n_drafted;
        }

        /////////////////////////////////////////Tree Decoding End///////////////////////////////////////

        // =========================================================================================
        // [추가] Token-level Reranking 알고리즘 (Verification 대상 토큰 축소)
        // 트리 전체의 생성된 토큰들을 누적 확률(Confidence Score) 기준으로 평가하여 Top-K 토큰만 남김
        // =========================================================================================
        int64_t step_rerank_us = 0;
        if (rerank) {
            const auto rerank_start = ggml_time_us();
            int total_drafted_tokens = batch_tgt.n_tokens - 1; // Root 토큰(index 0) 제외
            if (total_drafted_tokens > rerank_k) {
                // LOG_DBG("Token-Level Reranking: drafted tokens(%d) > rerank_k(%d), pruning tree...\n", total_drafted_tokens, rerank_k);

                struct TokenScore {
                    int t_idx;   // batch_tgt 내의 인덱스
                    float score; // 누적 확률 (Confidence Score)
                    int depth;   // 트리 깊이
                };

                std::vector<TokenScore> token_scores;
                // batch_tgt의 1번 인덱스부터는 Draft Model이 생성한 토큰들임
                for (int t = 1; t < batch_tgt.n_tokens; ++t) {
                    int depth = batch_tgt.pos[t] - n_past_tgt - 1; // drafting loop의 i 와 동일
                    int s = batch_tgt.seq_id[t][0]; // 토큰을 처음 생성했던 sequence ID
                    float score = scores[s][depth];
                    token_scores.push_back({t, score, depth});
                }

                // Score 기준 내림차순 정렬. 점수가 같으면 depth가 얕은(부모) 토큰을 우선하여 트리 무결성 철저히 보장
                std::sort(token_scores.begin(), token_scores.end(), [](const TokenScore& a, const TokenScore& b) {
                    if (a.score != b.score) return a.score > b.score;
                    return a.depth < b.depth;
                });

                // Top-K 토큰 인덱스 수집
                std::set<int> surviving_tokens;
                surviving_tokens.insert(0); // Root 토큰(프롬프트 마지막 토큰)은 무조건 유지
                for (int i = 0; i < rerank_k; ++i) {
                    surviving_tokens.insert(token_scores[i].t_idx);
                }

                // 3. Target Model의 연산량을 줄이기 위해 batch_tgt를 in-place로 압축
                int new_n_tokens = 0;
                std::vector<int> old_to_new_idx(batch_tgt.n_tokens, -1);

                for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                    if (surviving_tokens.count(t)) {
                        old_to_new_idx[t] = new_n_tokens;
                        batch_tgt.token[new_n_tokens]    = batch_tgt.token[t];
                        batch_tgt.pos[new_n_tokens]      = batch_tgt.pos[t];
                        batch_tgt.n_seq_id[new_n_tokens] = batch_tgt.n_seq_id[t];
                        for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                            batch_tgt.seq_id[new_n_tokens][p] = batch_tgt.seq_id[t][p];
                        }
                        batch_tgt.logits[new_n_tokens] = batch_tgt.logits[t];
                        new_n_tokens++;
                    }
                }

                // LOG_DBG("Token-Level Reranking: batch_tgt.n_tokens reduced from %d to %d\n", batch_tgt.n_tokens, new_n_tokens);
                batch_tgt.n_tokens = new_n_tokens;

                // 4. 잘려나간 토큰 정보 동기화 및 시퀀스 정리
                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].active) continue;

                    std::vector<int> new_i_batch_tgt;
                    std::vector<llama_token> new_tokens;
                    std::vector<std::vector<llama_token_data>> new_dists;

                    // resize()가 아닌 정확한 매핑으로 살아남은 토큰만 추출
                    for (size_t i = 0; i < drafts[s].i_batch_tgt.size(); ++i) {
                        int old_idx = drafts[s].i_batch_tgt[i];
                        if (old_idx >= 0 && old_idx < (int)old_to_new_idx.size() && old_to_new_idx[old_idx] != -1) {
                            new_i_batch_tgt.push_back(old_to_new_idx[old_idx]);
                            if (i < drafts[s].tokens.size()) {
                                new_tokens.push_back(drafts[s].tokens[i]);
                            }
                            if (i < drafts[s].dists.size()) {
                                new_dists.push_back(drafts[s].dists[i]);
                            }
                        }
                    }

                    // 시퀀스의 길이가 1 이하(루트 노드만 남음)라면 더 이상 Verification할 Draft 토큰이 없으므로 비활성화
                    if (new_i_batch_tgt.size() <= 1) {
                        drafts[s].active = false;

                        if (drafts[s].smpl != nullptr) {
                            common_sampler_free(drafts[s].smpl);
                            drafts[s].smpl = nullptr;
                        }

                        // 버려지는 시퀀스의 KV Cache를 Draft 메모리에서 즉시 삭제하여 슬롯 확보
                        llama_memory_seq_rm(mem_dft, s, -1, -1);
                    } else {
                        drafts[s].i_batch_tgt = new_i_batch_tgt;
                        drafts[s].tokens = new_tokens;
                        drafts[s].dists = new_dists;
                    }
                }

                // 5. [핵심 수정] batch_tgt의 seq_id 배열에서 비활성화된 시퀀스 ID 영구 제거
                for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                    int valid_seqs = 0;
                    for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                        int s = batch_tgt.seq_id[t][p];
                        // 메인 시퀀스(0)이거나 여전히 active 상태인 시퀀스만 남김
                        if (s != 0 && !drafts[s].active) {
                            continue;
                        }

                        bool already_present = false;
                        for (int q = 0; q < valid_seqs; ++q) {
                            if (batch_tgt.seq_id[t][q] == s) {
                                already_present = true;
                                break;
                            }
                        }

                        if (!already_present) {
                            batch_tgt.seq_id[t][valid_seqs++] = s;
                        }
                    }
                    batch_tgt.n_seq_id[t] = valid_seqs;
                }
            }
            const auto rerank_end = ggml_time_us();
            step_rerank_us = (rerank_end - rerank_start);
            total_tree_pruning_us += step_rerank_us;
        }
        // =========================================================================================

        /////////////////////////////////////////Drafting End///////////////////////////////////////

        const auto drafting_end = ggml_time_us();
        total_draft_wall_us += (drafting_end - drafting_start);
        double tree_decoding_latency = (drafting_end - drafting_start) / 1000.0;
        decoding_latencies.push_back(tree_decoding_latency);

        total_draft_tokens += batch_tgt.n_tokens - 1;

        if (draft_target_delay_ms > 0) {
            const auto delay_start = ggml_time_us();
            std::this_thread::sleep_for(std::chrono::milliseconds(draft_target_delay_ms));
            total_draft_target_delay_us += (ggml_time_us() - delay_start);
        }

        verification_start = ggml_time_us(); //verification 시작 시간 기록 -ym-

        // evaluate the target model on the drafted tokens using QNN
        {
            const auto step_target_forward_start = ggml_time_us();
            int64_t step_target_seq_refs = 0;
            int32_t step_target_shared_tokens = 0;
            int32_t step_target_max_seq_refs = 0;
            int64_t step_target_tree_attn_edges = 0;
            int64_t step_target_seq_compare_ops = 0;
            for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                const int32_t seq_refs = batch_tgt.n_seq_id[t];
                step_target_seq_refs += seq_refs;
                step_target_max_seq_refs = std::max(step_target_max_seq_refs, seq_refs);
                if (seq_refs > 1) {
                    ++step_target_shared_tokens;
                }
            }
            for (int i = 0; i < batch_tgt.n_tokens; ++i) {
                const int32_t p_i = batch_tgt.pos ? batch_tgt.pos[i] : i;
                for (int k = 0; k < batch_tgt.n_tokens; ++k) {
                    const int32_t p_k = batch_tgt.pos ? batch_tgt.pos[k] : k;
                    if (p_k > p_i) {
                        continue;
                    }

                    bool has_common_seq = false;
                    if (batch_tgt.n_seq_id && batch_tgt.seq_id) {
                        for (int si = 0; si < batch_tgt.n_seq_id[i] && !has_common_seq; ++si) {
                            for (int sk = 0; sk < batch_tgt.n_seq_id[k] && !has_common_seq; ++sk) {
                                ++step_target_seq_compare_ops;
                                if (batch_tgt.seq_id[i][si] == batch_tgt.seq_id[k][sk]) {
                                    has_common_seq = true;
                                }
                            }
                        }
                    } else {
                        has_common_seq = true;
                    }

                    if (has_common_seq) {
                        ++step_target_tree_attn_edges;
                    }
                }
            }
            total_target_batch_tokens += batch_tgt.n_tokens;
            total_target_batch_seq_refs += step_target_seq_refs;
            total_target_batch_shared_tokens += step_target_shared_tokens;
            total_target_tree_attn_edges += step_target_tree_attn_edges;
            total_target_seq_compare_ops += step_target_seq_compare_ops;
            max_target_batch_tokens = std::max(max_target_batch_tokens, batch_tgt.n_tokens);
            max_target_batch_seq_refs_per_token = std::max(max_target_batch_seq_refs_per_token, step_target_max_seq_refs);
            max_target_tree_attn_edges = std::max(max_target_tree_attn_edges, step_target_tree_attn_edges);
            max_target_seq_compare_ops = std::max(max_target_seq_compare_ops, step_target_seq_compare_ops);

            // QNN KV cache management: Copy seq 0 to all active sequences
            qnn_runner.kv_seq_keep(0);
            for (int s = 1; s < n_seq_dft; ++s) {
                // Reranking에서 살아남은(active) 시퀀스만 KV Cache 복사
                if (drafts[s].active) {
                    qnn_runner.kv_seq_cp(0, s, -1, -1);
                }
            }
            const auto target_kv_end = ggml_time_us();
            total_target_kv_cache_us += (target_kv_end - step_target_forward_start);

            ctx_tgt->final_hiddens.clear();  // Clear before verification decode to get fresh hidden states
            const auto t_dec_start = ggml_time_us();
            if (qnn_runner.qnn_decode(ctx_tgt, batch_tgt)) {
                LOG_ERR("%s: QNN verification decode failed: %s\n", __func__, qnn_runner.get_error().c_str());
                break;
            }
            const auto t_dec_end = ggml_time_us();
            // total_target_forward_us += (t_dec_end - t_dec_start);
            target_forward_us.push_back(t_dec_end - t_dec_start);
            const auto & target_prefill_profile = qnn_runner.get_last_multi_context_prefill_profile();
            total_target_slot_search_us += target_prefill_profile.slot_search_us;
            total_target_mask_build_us += target_prefill_profile.attn_mask_us;
            total_target_shard_prefill_us += target_prefill_profile.shard_prefill_us;
            total_target_shard_kv_override_us += target_prefill_profile.shard_kv_override_us;
            total_target_shard_input_fill_us += target_prefill_profile.shard_input_fill_us;
            total_target_shard_tensor_build_us += target_prefill_profile.shard_tensor_build_us;
            total_target_shard_execute_us += target_prefill_profile.shard_execute_us;
            total_target_shard_output_copy_us += target_prefill_profile.shard_output_copy_us;
            total_target_internal_kv_writeback_us += target_prefill_profile.kv_writeback_us;
            total_target_cell_meta_us += target_prefill_profile.cell_meta_us;
            total_target_logits_dequant_us += target_prefill_profile.logits_dequant_us;
            total_target_hidden_copy_us += target_prefill_profile.hidden_copy_us;
            total_target_logits_inject_us += target_prefill_profile.logits_inject_us;

            if (!ctx_tgt->final_hiddens.empty()) {
                backup_data = ctx_tgt->final_hiddens;  // Hidden states from QNN verification
            } else {
                fprintf(stderr, "[DIAG-VERIF] WARNING: final_hiddens empty after verification!\n");
            }

            for (int i = 0; i < n_seq_dft; i++) {
                temp_i_batch_dft[i] = 0;
            }
            ++n_past_tgt;
        }
        num_steps++;

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    {
        const double prefill_ms  = (t_enc_end - t_enc_start) / 1000.0;
        const double prefill_tps = n_input / (prefill_ms / 1000.0);
        const double decode_ms   = (t_dec_end - t_dec_start) / 1000.0;
        const double decode_tps  = n_predict / (decode_ms / 1000.0);
        const double decode_lat  = n_predict > 0 ? decode_ms / n_predict : 0;

        const int    n_steps     = (int)decoding_latencies.size();
        const double draft_len   = n_depth;
        const double draft_tokens_avg = n_steps > 0 ? (double)total_draft_tokens / n_steps : 0;
        const double accept_len  = (n_steps > 0 && acceptance_lengths.size() > 1)
            ? std::accumulate(acceptance_lengths.begin()+1, acceptance_lengths.end(), 0.0) / n_steps : 0;
        const double avg_draft_lat = !decoding_latencies.empty()
            ? std::accumulate(decoding_latencies.begin(), decoding_latencies.end(), 0.0) / decoding_latencies.size() : 0;
        const double avg_verify_lat = !verification_latencies.empty()
            ? std::accumulate(verification_latencies.begin(), verification_latencies.end(), 0.0) / verification_latencies.size() : 0;
        const double avg_td = !T_d.empty()
            ? std::accumulate(T_d.begin(), T_d.end(), 0.0) / T_d.size() : 0;

        const double avg_step_ms = avg_draft_lat + avg_verify_lat;
        int64_t total_target_forward_us = std::accumulate(target_forward_us.begin(), target_forward_us.end(), int64_t{0});
        const double avg_target_batch_tokens = num_steps > 0 ? (double) total_target_batch_tokens / num_steps : 0.0;
        const double avg_target_batch_seq_refs = num_steps > 0 ? (double) total_target_batch_seq_refs / num_steps : 0.0;
        const double avg_target_batch_shared_tokens = num_steps > 0 ? (double) total_target_batch_shared_tokens / num_steps : 0.0;
        const double avg_target_tree_attn_edges = num_steps > 0 ? (double) total_target_tree_attn_edges / num_steps : 0.0;
        const double avg_target_seq_compare_ops = num_steps > 0 ? (double) total_target_seq_compare_ops / num_steps : 0.0;
        const double avg_target_batch_seq_refs_per_token = total_target_batch_tokens > 0
            ? (double) total_target_batch_seq_refs / total_target_batch_tokens
            : 0.0;
        const int64_t total_verification_accounted_us =
            total_target_forward_us + total_target_kv_cache_us + total_verify_logic_us + total_fallback_sampling_us;
        const int64_t total_verification_unaccounted_us = std::max<int64_t>(
            0,
            total_verification_wall_us - total_verification_accounted_us);
        const int64_t fallback_sampler_other_us = std::max<int64_t>(
                0,
                total_fallback_sampling_us - total_fallback_sampler_profile.sample_total_us);
        const int64_t fallback_sample_calls = total_fallback_sampler_profile.sample_calls;

        const auto avg_ms_per = [](int64_t total_us, int64_t count) -> double {
            return count > 0 ? static_cast<double>(total_us) / 1000.0 / static_cast<double>(count) : 0.0;
        };
        const auto avg_per = [](int64_t total, int64_t count) -> double {
            return count > 0 ? static_cast<double>(total) / static_cast<double>(count) : 0.0;
        };
        const int64_t total_expansion_split_us =
                total_split_kv_copy_us + total_split_history_update_us + total_split_draft_state_alloc_us;
        const int64_t total_expansion_us =
                total_expansion_loop_prep_us +
                total_expansion_sampling_us +
                total_expansion_candidate_bookkeeping_us +
                total_expansion_split_us +
                total_expansion_temp_probs_us +
                total_expansion_topk_us +
                total_expansion_target_batch_us +
                total_expansion_post_batch_us;
        const int64_t total_draft_accounted_us =
                total_draft_recompute_us +
                total_draft_setup_us +
                total_expansion_us +
                total_draft_forward_us +
                total_tree_pruning_us;
        const int64_t total_draft_unaccounted_us = std::max<int64_t>(0, total_draft_wall_us - total_draft_accounted_us);

        LOG_INF("\n");
        LOG_INF("======= Latency Breakdown (Avg / Step, Additive) =========\n");
        LOG_INF("Prefill (one-time)              : %8.3f ms\n", prefill_ms);
        LOG_INF("Decode steps                    : %8d\n", num_steps);
        LOG_INF("Fallback sample calls           : %8lld\n", (long long) fallback_sample_calls);

        LOG_INF("[1] Drafting Phase (top-level items sum to Avg draft phase)\n");
        LOG_INF("  - Draft Recompute/Alignment   : %8.3f ms\n", avg_ms_per(total_draft_recompute_us, num_steps));
        LOG_INF("    * KV/State Reset            : %8.3f ms\n", avg_ms_per(total_draft_recompute_state_reset_us, num_steps));
        LOG_INF("    * Replay Accepted Tokens    : %8.3f ms\n", avg_ms_per(total_draft_recompute_replay_decode_us, num_steps));
        LOG_INF("    * Current Token Re-decode   : %8.3f ms\n", avg_ms_per(total_draft_recompute_current_decode_us, num_steps));
        LOG_INF("    * Recompute steps           : %8.3f / step (%lld total steps)\n",
                avg_per(total_draft_recompute_steps, num_steps),
                (long long) total_draft_recompute_steps);
        LOG_INF("    * Replay decode steps       : %8.3f / step (%lld total steps)\n",
                avg_per(total_draft_recompute_replay_steps, num_steps),
                (long long) total_draft_recompute_replay_steps);
        LOG_INF("  - Draft Phase Setup           : %8.3f ms\n", avg_ms_per(total_draft_setup_us, num_steps));
        LOG_INF("  - Tree Expansion Total        : %8.3f ms\n", avg_ms_per(total_expansion_us, num_steps));
        LOG_INF("    * Loop Prep                 : %8.3f ms\n", avg_ms_per(total_expansion_loop_prep_us, num_steps));
        LOG_INF("    * Sampling from Draft       : %8.3f ms\n", avg_ms_per(total_expansion_sampling_us, num_steps));
        LOG_INF("    * Candidate Bookkeeping     : %8.3f ms\n", avg_ms_per(total_expansion_candidate_bookkeeping_us, num_steps));
        LOG_INF("    * Split Sequence            : %8.3f ms\n", avg_ms_per(total_expansion_split_us, num_steps));
        LOG_INF("      - KV Cache Copy           : %8.3f ms\n", avg_ms_per(total_split_kv_copy_us, num_steps));
        LOG_INF("      - Seq History Update      : %8.3f ms\n", avg_ms_per(total_split_history_update_us, num_steps));
        LOG_INF("      - Draft State Alloc       : %8.3f ms\n", avg_ms_per(total_split_draft_state_alloc_us, num_steps));
        LOG_INF("    * Temp Probs Array Prep     : %8.3f ms\n", avg_ms_per(total_expansion_temp_probs_us, num_steps));
        LOG_INF("    * TopK Sorting              : %8.3f ms\n", avg_ms_per(total_expansion_topk_us, num_steps));
        LOG_INF("    * Target Batch Append       : %8.3f ms\n", avg_ms_per(total_expansion_target_batch_us, num_steps));
        LOG_INF("    * Post Batch Bookkeeping    : %8.3f ms\n", avg_ms_per(total_expansion_post_batch_us, num_steps));
        LOG_INF("  - Draft Tree Forward          : %8.3f ms\n", avg_ms_per(total_draft_forward_us, num_steps));
        LOG_INF("    * Pre-sync submit           : %8.3f ms\n", avg_ms_per(total_draft_forward_submit_us, num_steps));
        LOG_INF("    * Post-submit sync wait     : %8.3f ms\n", avg_ms_per(total_draft_forward_sync_us, num_steps));
        LOG_INF("    * Tree decode calls         : %8.3f / step (%d total)\n",
                avg_per(total_draft_forward_calls, num_steps), total_draft_forward_calls);
        {
            const auto ed = llama_perf_eagle_draft(ctx_dft);
            const int32_t n_calls_all = ed.n_decode_calls;
            const double calls_per_step = num_steps > 0 ? (double) n_calls_all / (double) num_steps : 0.0;
            LOG_INF("  - Eagle Decode Breakdown (cumulative across ALL llama_decode_eagle calls: recompute + tree)\n");
            LOG_INF("    * Total decode calls        : %d (%.3f / step)\n", n_calls_all, calls_per_step);
            LOG_INF("    * Graph cache hits (reused) : %d  (%.1f%%)\n",
                    ed.n_graph_reused, n_calls_all > 0 ? 100.0 * ed.n_graph_reused / n_calls_all : 0.0);
            LOG_INF("    * Graph rebuilds            : %d  (%.1f%%)\n",
                    ed.n_graph_rebuilt, n_calls_all > 0 ? 100.0 * ed.n_graph_rebuilt / n_calls_all : 0.0);
            LOG_INF("    * Avg per call: apply_mctx  : %8.3f ms\n", n_calls_all > 0 ? (ed.t_apply_mctx_us    / 1000.0) / n_calls_all : 0.0);
            LOG_INF("    * Avg per call: graph_build : %8.3f ms (amortized; actual on rebuild only)\n",
                    n_calls_all > 0 ? (ed.t_graph_build_us   / 1000.0) / n_calls_all : 0.0);
            if (ed.n_graph_rebuilt > 0) {
                LOG_INF("      - Per rebuild             : %8.3f ms\n", (ed.t_graph_build_us / 1000.0) / ed.n_graph_rebuilt);
            }
            LOG_INF("    * Avg per call: set_inputs  : %8.3f ms\n", n_calls_all > 0 ? (ed.t_set_inputs_us    / 1000.0) / n_calls_all : 0.0);
            LOG_INF("    * Avg per call: graph_compute: %7.3f ms (submit + any in-call wait)\n",
                    n_calls_all > 0 ? (ed.t_graph_compute_us / 1000.0) / n_calls_all : 0.0);
            LOG_INF("    * Avg per step (all phases) : %8.3f ms\n",
                    num_steps > 0 ? ((ed.t_apply_mctx_us + ed.t_graph_build_us + ed.t_set_inputs_us + ed.t_graph_compute_us) / 1000.0) / num_steps : 0.0);
        }
        LOG_INF("  - Tree Pruning (Reranking)    : %8.3f ms\n", avg_ms_per(total_tree_pruning_us, num_steps));
        LOG_INF("  - Draft Residual              : %8.3f ms\n", avg_ms_per(total_draft_unaccounted_us, num_steps));
        LOG_INF("[1a] Draft Depth Latency (avg / visit)\n");
        for (int depth = 0; depth < n_depth; ++depth) {
            if (per_depth_visit_counts[depth] == 0) {
                continue;
            }
            LOG_INF("  - Depth %d Total              : %8.3f ms (%lld visits)\n",
                    depth,
                    avg_ms_per(per_depth_total_us[depth], per_depth_visit_counts[depth]),
                    (long long) per_depth_visit_counts[depth]);
            LOG_INF("    * Pre-forward Work          : %8.3f ms / visit\n",
                    avg_ms_per(per_depth_body_us[depth], per_depth_visit_counts[depth]));
            LOG_INF("    * Draft Forward             : %8.3f ms / visit\n",
                    avg_ms_per(per_depth_forward_us[depth], per_depth_visit_counts[depth]));
            LOG_INF("    * Forward Calls             : %8.3f / visit (%lld total)\n",
                    avg_per(per_depth_forward_calls[depth], per_depth_visit_counts[depth]),
                    (long long) per_depth_forward_calls[depth]);
        }

        LOG_INF("[2] Verification Phase (avg / step)\n");
        LOG_INF("  - Target Model Forward        : %8.3f ms\n", avg_ms_per(total_target_forward_us, num_steps));
        LOG_INF("  - Target KV Cache Management  : %8.3f ms\n", avg_ms_per(total_target_kv_cache_us, num_steps));
        LOG_INF("  - Target Batch Tokens         : %8.3f / step (max %d)\n",
                avg_target_batch_tokens,
                max_target_batch_tokens);
        LOG_INF("  - Target Batch SeqRefs        : %8.3f / step (%8.3f / tok, max %d)\n",
                avg_target_batch_seq_refs,
                avg_target_batch_seq_refs_per_token,
                max_target_batch_seq_refs_per_token);
        LOG_INF("  - Target Shared Tokens        : %8.3f / step\n", avg_target_batch_shared_tokens);
        LOG_INF("  - Target Tree Attn Edges      : %8.3f / step (max %lld)\n",
                avg_target_tree_attn_edges,
                (long long) max_target_tree_attn_edges);
        LOG_INF("  - Target Seq Compare Ops      : %8.3f / step (max %lld)\n",
                avg_target_seq_compare_ops,
                (long long) max_target_seq_compare_ops);
        LOG_INF("  - Target Slot Search          : %8.3f ms\n", avg_ms_per(total_target_slot_search_us, num_steps));
        LOG_INF("  - Target Mask Build           : %8.3f ms\n", avg_ms_per(total_target_mask_build_us, num_steps));
        LOG_INF("  - Target Shard Prefill        : %8.3f ms\n", avg_ms_per(total_target_shard_prefill_us, num_steps));
        LOG_INF("    * Shard KV Override         : %8.3f ms\n", avg_ms_per(total_target_shard_kv_override_us, num_steps));
        LOG_INF("    * Shard Input Fill          : %8.3f ms\n", avg_ms_per(total_target_shard_input_fill_us, num_steps));
        LOG_INF("    * Shard Tensor Build        : %8.3f ms\n", avg_ms_per(total_target_shard_tensor_build_us, num_steps));
        LOG_INF("    * Shard Execute             : %8.3f ms\n", avg_ms_per(total_target_shard_execute_us, num_steps));
        LOG_INF("    * Shard Output Copy         : %8.3f ms\n", avg_ms_per(total_target_shard_output_copy_us, num_steps));
        LOG_INF("  - Target Internal KV Write    : %8.3f ms\n", avg_ms_per(total_target_internal_kv_writeback_us, num_steps));
        LOG_INF("  - Target Cell Meta Update     : %8.3f ms\n", avg_ms_per(total_target_cell_meta_us, num_steps));
        LOG_INF("  - Target Logits Dequant       : %8.3f ms\n", avg_ms_per(total_target_logits_dequant_us, num_steps));
        LOG_INF("  - Target Hidden Copy          : %8.3f ms\n", avg_ms_per(total_target_hidden_copy_us, num_steps));
        LOG_INF("  - Target Logits Inject        : %8.3f ms\n", avg_ms_per(total_target_logits_inject_us, num_steps));
        LOG_INF("  - Tree Verification Logic     : %8.3f ms\n", avg_ms_per(total_verify_logic_us, num_steps));
        LOG_INF("  - Fallback Sampling           : %8.3f ms\n", avg_ms_per(total_fallback_sampling_us, num_steps));
        LOG_INF("  - Verification Unaccounted    : %8.3f ms\n", avg_ms_per(total_verification_unaccounted_us, num_steps));
        if (fallback_sample_calls > 0) {
            LOG_INF("    * Sampler sync              : %8.3f ms / call\n",
                    avg_ms_per(total_fallback_sampler_profile.sync_us, fallback_sample_calls));
            LOG_INF("    * set_logits                : %8.3f ms / call\n",
                    avg_ms_per(total_fallback_sampler_profile.set_logits_us, fallback_sample_calls));
            LOG_INF("      - logits fetch            : %8.3f ms / call\n",
                    avg_ms_per(total_fallback_sampler_profile.get_logits_us, fallback_sample_calls));
            LOG_INF("      - candidate build         : %8.3f ms / call\n",
                    avg_ms_per(total_fallback_sampler_profile.build_candidates_us, fallback_sample_calls));
            LOG_INF("    * Grammar apply/check       : %8.3f ms / call\n",
                    avg_ms_per(total_fallback_sampler_profile.grammar_apply_us + total_fallback_sampler_profile.grammar_check_us, fallback_sample_calls));
            LOG_INF("    * Sampler chain apply       : %8.3f ms / call\n",
                    avg_ms_per(total_fallback_sampler_profile.chain_apply_us, fallback_sample_calls));
            LOG_INF("    * Resample passes           : %8.3f / call\n",
                    avg_per(total_fallback_sampler_profile.resample_count, fallback_sample_calls));
            LOG_INF("    * Non-sampler overhead      : %8.3f ms / call\n",
                    avg_ms_per(fallback_sampler_other_us, fallback_sample_calls));
        }

        LOG_INF("[3] Injected Handoff Delay (avg / step)\n");
        LOG_INF("  - Draft -> Target Delay       : %8.3f ms\n", avg_ms_per(total_draft_target_delay_us, num_steps));
        LOG_INF("  - Target -> Draft Delay       : %8.3f ms\n", avg_ms_per(total_target_draft_delay_us, num_steps));
        LOG_INF("=====================================================\n");
        
        fprintf(stderr, "\n");
        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "          EAGLE-2-QNN  Performance Summary\n");
        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "  Prefill           : %5d tokens | %9.2f ms | %8.2f t/s\n", n_input, prefill_ms, prefill_tps);
        fprintf(stderr, "  Decode            : %5d tokens | %9.2f ms | %8.2f t/s\n", n_predict, decode_ms, decode_tps);
        fprintf(stderr, "  Decode latency    :              | %9.2f ms/tok\n", decode_lat);
        fprintf(stderr, "------------------------------------------------------------\n");
        fprintf(stderr, "  n_draft   = %d\n", n_draft);
        fprintf(stderr, "  n_predict = %d\n", n_predict);
        fprintf(stderr, "  n_drafted = %d (depth), total_draft_tokens = %d\n", n_drafted, total_draft_tokens);
        fprintf(stderr, "  n_accept  = %d\n", n_accept);
        fprintf(stderr, "  Draft length (depth)  : %.3f\n", draft_len);
        fprintf(stderr, "  Draft tokens/step     : %.3f\n", draft_tokens_avg);
        fprintf(stderr, "  Avg accept length     : %.3f\n", accept_len);
        fprintf(stderr, "  Accept ratio          : %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
        fprintf(stderr, "------------------------------------------------------------\n");
        fprintf(stderr, "  Avg draft phase       : %9.3f ms\n", avg_draft_lat);
        fprintf(stderr, "  Avg verification      : %9.3f ms\n", avg_verify_lat);
        fprintf(stderr, "  Avg T_d (1-tok dft)   : %9.3f ms\n", avg_td);
        fprintf(stderr, "  Avg (draft+verify)    : %9.3f ms\n", avg_step_ms);
        fprintf(stderr, "  Gap draft->target     : %9.3f ms / step\n", avg_ms_per(total_draft_target_delay_us, num_steps));
        fprintf(stderr, "  Gap target->draft     : %9.3f ms / step\n", avg_ms_per(total_target_draft_delay_us, num_steps));
        fprintf(stderr, "------------------------------------------------------------\n");
    }

    // [추가] 수락 길이 통계 계산 및 출력
    if (acceptance_lengths.size() > 1) {
        const double avg_len = std::accumulate(acceptance_lengths.begin()+1, acceptance_lengths.end(), 0.0) / (acceptance_lengths.size()-1);
        const int min_len = *std::min_element(acceptance_lengths.begin()+1, acceptance_lengths.end());
        const int max_len = *std::max_element(acceptance_lengths.begin()+1, acceptance_lengths.end());

        LOG_INF("\n");
        LOG_INF("Acceptance length stats:\n");
        LOG_INF("  Min length: %d\n", min_len);
        LOG_INF("  Max length: %d\n", max_len);
        LOG_INF("  Avg length: %.3f\n", avg_len);
    }

    if (!decoding_latencies.empty() && !verification_latencies.empty()) {
        const double avg_decoding_latency = std::accumulate(decoding_latencies.begin(), decoding_latencies.end(), 0.0) / decoding_latencies.size();
        const double avg_verification_latency = std::accumulate(verification_latencies.begin(), verification_latencies.end(), 0.0) / verification_latencies.size();
        LOG_INF("\navg drafting latency: %.3f ms\n", avg_decoding_latency);
        LOG_INF("avg verification latency: %.3f ms\n", avg_verification_latency);
        LOG_INF("avg T_d: %.3f ms\n", !T_d.empty() ? (std::accumulate(T_d.begin(), T_d.end(), 0.0) / T_d.size()) : 0.0);
        LOG_INF("Verification/Draft Phase Count: %zu\n", verification_latencies.size());
    }

    // // Accepted Token Counts Matrix 출력 (디버깅용)
    // LOG_INF("\nAccepted Token Counts Matrix:\n");
    // for (int i = 0; i < rows; i++) {
    //     LOG_INF("[");
    //     for (int j = 0; j < cols; j++) {
    //         // 기존 "%d " 대신 "%3d"를 사용하여 너비를 3으로 맞춥니다.
    //         LOG_INF("%3d", accept_counts[i][j]);
    //     }
    //     LOG_INF(" ]\n");
    // }

    // Save data files
    {
        std::ofstream f1("al_d25.txt");
        if (f1.is_open()) { for (auto v : acceptance_lengths) f1 << v << "\n"; }
        std::ofstream f2("cs_d25.txt");
        if (f2.is_open()) { for (auto v : confidence_scores) f2 << v << "\n"; }
        std::ofstream f3("vl_d25.txt");
        if (f3.is_open()) { for (auto v : verification_latencies) f3 << v << "\n"; }
        std::ofstream f4("dr_d25.txt");
        if (f4.is_open()) { for (auto v : decoding_latencies) f4 << v << "\n"; }
        std::ofstream f5("var_for.txt");
        if (f5.is_open()) { for (auto v : target_forward_us) f5 << v << "\n"; }
    }

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
