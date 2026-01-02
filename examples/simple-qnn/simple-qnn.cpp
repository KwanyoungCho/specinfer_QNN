#include "llama.h"
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "QNN/llm_decode_runner.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    ./llama-simple-qnn --qnn -n 512 --multi-context --ctx-dir ctx_out --tokenizer ctx_out/tokenizer.gguf --params ctx_out/params.json --log-level 1 -f ../../gguf/prompt.txt \n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    // prompt and generation length (can be overridden by CLI params)
    std::string prompt;
    int n_predict = 0;

    // parse command line arguments
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN)) {
        return 1;
    }
    common_init();

    // derive prompt and n_predict from params with sensible defaults
    prompt = params.prompt.empty() ? "Hello my name is" : params.prompt;
    n_predict = params.n_predict > 0 ? params.n_predict : 100;

    // QNN config (optional)
    llama_qnn::LLMDecodeConfig qnn_config;
    if (params.use_qnn) {
        qnn_config.ctx_dir           = params.qnn_ctx_dir;
        qnn_config.backend_so        = params.qnn_backend_so;
        qnn_config.system_so         = params.qnn_system_so;
        qnn_config.tokenizer_path    = params.qnn_tokenizer_path;
        qnn_config.params_path       = params.qnn_params_path;
        qnn_config.max_gen_tokens    = n_predict;
        qnn_config.log_level         = params.qnn_log_level;
        qnn_config.use_multi_context = params.qnn_use_multi_context;
        qnn_config.num_shards        = params.qnn_num_shards;

        if (qnn_config.ctx_dir.empty() || qnn_config.tokenizer_path.empty()) {
            fprintf(stderr, "Error: QNN required paths not specified (ctx_dir/tokenizer).\n");
            return 1;
        }
        if (qnn_config.backend_so.empty() || qnn_config.system_so.empty()) {
            qnn_config.backend_so = "libQnnHtp.so";
            qnn_config.system_so = "libQnnSystem.so";
        }
    }

    // ggml_backend_load_all();

    llama_model_params tokenizer_param = llama_model_default_params();
    tokenizer_param.vocab_only = true;

    llama_model * model = llama_model_load_from_file(qnn_config.tokenizer_path.c_str(), tokenizer_param);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt (used only for llama_decode path)

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // If QNN is enabled, run generation via QNN using prompt_tokens
    if (params.use_qnn) {
        const auto t_init_start = ggml_time_us();
        llama_qnn::LLMDecodeRunner runner(qnn_config);
        if (!runner.initialize()) {
            fprintf(stderr, "Error: Failed to initialize QNN runner: %s\n", runner.get_error().c_str());
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        const auto t_init_end = ggml_time_us();
        printf("[QNN] Initialize time: %f ms\n", (t_init_end - t_init_start) / 1000.0); 

        // QNN Prefill: use qnn_decode with batch (same interface as llama_decode)
        const auto t_inference_start = ggml_time_us();
        llama_batch prefill_batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        
        if (runner.qnn_decode(ctx, prefill_batch)) {
            fprintf(stderr, "[QNN] qnn_decode (prefill) failed: %s\n", runner.get_error().c_str());
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        const auto t_inference_end = ggml_time_us();
        printf("[QNN] Prefill time: %f ms\n", (t_inference_end - t_inference_start) / 1000.0);

        // First token sampling using llama sampler on QNN-provided logits
        llama_token cur_token = llama_sampler_sample(smpl, ctx, -1);

        // print first generated token
        {
            char buf[128];
            int n = llama_token_to_piece(vocab, cur_token, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                llama_sampler_free(smpl);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);
        }

        // Decode loop using qnn_decode with batch (same interface as llama_decode)
        int n_decode_qnn = 1;

        for (int i = 0; i < n_predict - 1; ++i) {
            // Create batch with single token (same as llama_decode usage)
            llama_batch decode_batch = llama_batch_get_one(&cur_token, 1);
            
            if (runner.qnn_decode(ctx, decode_batch)) {
                fprintf(stderr, "[QNN] qnn_decode (decode) failed: %s\n", runner.get_error().c_str());
                break;
            }

            // sample next token from llama sampler using QNN-provided logits
            cur_token = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab, cur_token)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, cur_token, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                llama_sampler_free(smpl);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            n_decode_qnn += 1;
        }

        printf("\n");

        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);

        return 0;
    }

    // main loop (fallback to llama_decode when QNN is not used)

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
