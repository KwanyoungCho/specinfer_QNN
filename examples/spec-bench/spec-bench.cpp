// Spec-Bench runner for EAGLE speculative decoding
// Based on speculative-eagle.cpp — loads models once, runs multiple prompts sequentially.
//
// Usage:
//   ./llama-spec-bench --bench-file prompts.jsonl [all normal speculative-eagle flags]
//   ./llama-spec-bench --bench-file prompts.jsonl -n 512 -m model.gguf -md draft.gguf ...
//
// JSONL format (Spec-Bench):
//   {"question_id": 81, "category": "writing", "turns": ["prompt text..."]}
//
// Also accepts plain-text files (one prompt per line).

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <unordered_map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

#define n_depth 5
#define expand_k 2
#define rerank_k 10

// ============================================================
// Structs
// ============================================================

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

    auto * cb_data = (struct callback_data *) user_data;
    auto n_bytes = ggml_nbytes(tensor);
    size_t prev_size = cb_data->data.size();
    cb_data->data.resize(prev_size + n_bytes / sizeof(float));
    ggml_backend_tensor_get(tensor, cb_data->data.data() + prev_size, 0, n_bytes);
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

// ============================================================
// Per-prompt result
// ============================================================

struct bench_result {
    int    question_id;
    std::string category;
    int    n_input;
    int    n_predict;
    int    n_drafted;
    int    n_accept;
    double prefill_ms;
    double decode_ms;
    double prefill_tps;
    double decode_tps;
    double decode_lat;
    double avg_accept_len;
    double accept_ratio;
    double avg_draft_lat;
    double avg_verify_lat;
    double avg_td;
    bool   success;
    std::string output_text;
};

// ============================================================
// Token frequency tracking for vocab compression analysis
// ============================================================

struct token_freq_stats {
    std::unordered_map<llama_token, int64_t> draft_freq;
    std::unordered_map<llama_token, int64_t> draft_accepted;
    std::unordered_map<llama_token, int64_t> bonus_freq;
};

// ============================================================
// JSONL prompt loader
// ============================================================

struct bench_prompt {
    int question_id;
    std::string category;
    std::string text;
};

static std::string json_get_string(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++;
    size_t end = pos;
    while (end < json.size() && json[end] != '"') {
        if (json[end] == '\\') end++; // skip escaped char
        end++;
    }
    return json.substr(pos, end - pos);
}

static int json_get_int(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return -1;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return -1;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return std::atoi(json.c_str() + pos);
}

static std::string json_get_first_turn(const std::string & json) {
    size_t pos = json.find("\"turns\"");
    if (pos == std::string::npos) return "";
    pos = json.find('[', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    pos++;
    std::string result;
    while (pos < json.size()) {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            char c = json[pos + 1];
            if (c == '"') result += '"';
            else if (c == 'n') result += '\n';
            else if (c == 't') result += '\t';
            else if (c == '\\') result += '\\';
            else { result += '\\'; result += c; }
            pos += 2;
        } else if (json[pos] == '"') {
            break;
        } else {
            result += json[pos];
            pos++;
        }
    }
    return result;
}

static std::vector<bench_prompt> load_prompts(const std::string & path) {
    std::vector<bench_prompt> prompts;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "Error: cannot open bench file: %s\n", path.c_str());
        return prompts;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(ifs, line)) {
        line_num++;
        if (line.empty()) continue;

        bench_prompt p;
        if (line[0] == '{') {
            // JSONL format
            p.question_id = json_get_int(line, "question_id");
            p.category = json_get_string(line, "category");
            p.text = json_get_first_turn(line);
        } else {
            // plain text: one prompt per line
            p.question_id = line_num;
            p.category = "plain";
            p.text = line;
        }

        if (!p.text.empty()) {
            prompts.push_back(std::move(p));
        }
    }

    return prompts;
}

// ============================================================
// ShareGPT JSON loader — extracts all "from":"human" turns
// ============================================================

static void sg_skip_ws(const std::string & s, size_t & pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) pos++;
}

static std::string sg_parse_string(const std::string & s, size_t & pos) {
    if (pos >= s.size() || s[pos] != '"') return "";
    pos++;
    std::string result;
    while (pos < s.size()) {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            char c = s[pos + 1];
            switch (c) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                default:   result += '\\'; result += c; break;
            }
            pos += 2;
        } else if (s[pos] == '"') {
            pos++;
            return result;
        } else {
            result += s[pos];
            pos++;
        }
    }
    return result;
}

static void sg_skip_value(const std::string & s, size_t & pos) {
    sg_skip_ws(s, pos);
    if (pos >= s.size()) return;
    if (s[pos] == '"') {
        sg_parse_string(s, pos);
    } else if (s[pos] == '{' || s[pos] == '[') {
        char open = s[pos], close = (open == '{') ? '}' : ']';
        int depth = 1; pos++;
        bool in_str = false;
        while (pos < s.size() && depth > 0) {
            if (in_str) {
                if (s[pos] == '\\') { pos += 2; continue; }
                if (s[pos] == '"') in_str = false;
            } else {
                if (s[pos] == '"') in_str = true;
                else if (s[pos] == open) depth++;
                else if (s[pos] == close) depth--;
            }
            pos++;
        }
    } else {
        while (pos < s.size() && s[pos] != ',' && s[pos] != ']' && s[pos] != '}') pos++;
    }
}

static std::vector<bench_prompt> load_sharegpt_prompts(const std::string & path) {
    std::vector<bench_prompt> prompts;

    fprintf(stderr, "[ShareGPT] Loading %s ...\n", path.c_str());
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        fprintf(stderr, "Error: cannot open file: %s\n", path.c_str());
        return prompts;
    }
    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    fprintf(stderr, "[ShareGPT] File loaded (%.1f MB), parsing...\n", s.size() / (1024.0 * 1024.0));

    size_t pos = 0;
    sg_skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') {
        fprintf(stderr, "Error: expected JSON array\n");
        return prompts;
    }
    pos++;

    int prompt_id = 0;
    int conv_count = 0;

    while (pos < s.size()) {
        sg_skip_ws(s, pos);
        if (pos >= s.size() || s[pos] == ']') break;
        if (s[pos] == ',') { pos++; continue; }
        if (s[pos] != '{') break;
        pos++;

        std::string conv_id;
        conv_count++;

        while (pos < s.size()) {
            sg_skip_ws(s, pos);
            if (s[pos] == '}') { pos++; break; }
            if (s[pos] == ',') { pos++; continue; }

            std::string key = sg_parse_string(s, pos);
            sg_skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ':') pos++;
            sg_skip_ws(s, pos);

            if (key == "id") {
                conv_id = sg_parse_string(s, pos);
            } else if (key == "conversations") {
                if (pos >= s.size() || s[pos] != '[') { sg_skip_value(s, pos); continue; }
                pos++;

                while (pos < s.size()) {
                    sg_skip_ws(s, pos);
                    if (s[pos] == ']') { pos++; break; }
                    if (s[pos] == ',') { pos++; continue; }
                    if (s[pos] != '{') break;
                    pos++;

                    std::string from_val, value_val;

                    while (pos < s.size()) {
                        sg_skip_ws(s, pos);
                        if (s[pos] == '}') { pos++; break; }
                        if (s[pos] == ',') { pos++; continue; }

                        std::string ckey = sg_parse_string(s, pos);
                        sg_skip_ws(s, pos);
                        if (pos < s.size() && s[pos] == ':') pos++;
                        sg_skip_ws(s, pos);

                        if (ckey == "from") {
                            from_val = sg_parse_string(s, pos);
                        } else if (ckey == "value") {
                            value_val = sg_parse_string(s, pos);
                        } else {
                            sg_skip_value(s, pos);
                        }
                    }

                    if (from_val == "human" && !value_val.empty()) {
                        bench_prompt p;
                        p.question_id = ++prompt_id;
                        p.category = conv_id;
                        p.text = std::move(value_val);
                        prompts.push_back(std::move(p));
                    }
                }
            } else {
                sg_skip_value(s, pos);
            }
        }

        if (conv_count % 10000 == 0) {
            fprintf(stderr, "[ShareGPT] Parsed %d conversations, %d human turns so far...\n", conv_count, prompt_id);
        }
    }

    fprintf(stderr, "[ShareGPT] Done: %d conversations, %d human turns extracted\n", conv_count, prompt_id);
    return prompts;
}

static std::string format_llama3_prompt(const std::string & user_msg) {
    return
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully "
        "as possible, while being safe.  Your answers should not include any harmful, "
        "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
        "that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why "
        "instead of answering something not correct. If you don't know the answer to a "
        "question, please don't share false information.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + user_msg +
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";
}

static std::string format_vicuna_prompt(const std::string & user_msg) {
    return
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: " + user_msg + " ASSISTANT:";
}

static std::string apply_template(const std::string & tmpl, const std::string & user_msg) {
    if (tmpl == "llama3") return format_llama3_prompt(user_msg);
    if (tmpl == "vicuna") return format_vicuna_prompt(user_msg);
    return user_msg;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char ** argv) {

    // ------ Extract custom args before common_params_parse ------
    std::string bench_file;
    std::string chat_template = "llama3";  // default
    std::string dataset_type  = "auto";    // auto, specbench, sharegpt
    std::vector<char *> filtered_argv;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench-file" && i + 1 < argc) {
            bench_file = argv[++i];
        } else if (std::string(argv[i]) == "--chat-template" && i + 1 < argc) {
            chat_template = argv[++i];
        } else if (std::string(argv[i]) == "--no-chat-template") {
            chat_template = "none";
        } else if (std::string(argv[i]) == "--dataset-type" && i + 1 < argc) {
            dataset_type = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    if (bench_file.empty()) {
        fprintf(stderr, "Usage: %s --bench-file <prompts.jsonl> [speculative-eagle args...]\n", argv[0]);
        fprintf(stderr, "\nRun EAGLE speculative decoding on multiple prompts (models loaded once).\n");
        fprintf(stderr, "  --bench-file FILE        JSONL file with prompts (Spec-Bench format) or plain text\n");
        fprintf(stderr, "  --chat-template TMPL     Chat template: llama3 (default), vicuna, none\n");
        fprintf(stderr, "  --no-chat-template       Same as --chat-template none\n");
        fprintf(stderr, "  --dataset-type TYPE      Dataset format: auto (default), specbench, sharegpt\n");
        return 1;
    }

    if (dataset_type != "auto" && dataset_type != "specbench" && dataset_type != "sharegpt") {
        fprintf(stderr, "Error: unknown dataset type '%s'. Use: auto, specbench, sharegpt\n", dataset_type.c_str());
        return 1;
    }

    if (chat_template != "llama3" && chat_template != "vicuna" && chat_template != "none") {
        fprintf(stderr, "Error: unknown chat template '%s'. Use: llama3, vicuna, none\n", chat_template.c_str());
        return 1;
    }

    int new_argc = (int)filtered_argv.size();
    char ** new_argv = filtered_argv.data();

    common_params params;
    params.sampling.n_probs = 128;

    if (!common_params_parse(new_argc, new_argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        fprintf(stderr, "Error: --n-predict must be >= -1\n");
        return 1;
    }

    common_init();

    if (params.speculative.model.path.empty()) {
        fprintf(stderr, "Error: --model-draft is required\n");
        return 1;
    }

    // ------ Load prompts ------
    if (dataset_type == "auto") {
        size_t dot = bench_file.rfind('.');
        std::string ext = (dot != std::string::npos) ? bench_file.substr(dot) : "";
        if (ext == ".json") {
            dataset_type = "sharegpt";
        } else {
            dataset_type = "specbench";
        }
        fprintf(stderr, "[Spec-Bench] Auto-detected dataset type: %s\n", dataset_type.c_str());
    }

    std::vector<bench_prompt> prompts;
    if (dataset_type == "sharegpt") {
        prompts = load_sharegpt_prompts(bench_file);
    } else {
        prompts = load_prompts(bench_file);
    }

    if (prompts.empty()) {
        fprintf(stderr, "Error: no prompts loaded from %s\n", bench_file.c_str());
        return 1;
    }
    fprintf(stderr, "[Spec-Bench] Loaded %zu prompts from %s (type: %s, template: %s)\n",
            prompts.size(), bench_file.c_str(), dataset_type.c_str(), chat_template.c_str());

    const int n_seq_dft = params.n_parallel;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // ====================================================================
    // Model loading (ONE TIME)
    // ====================================================================
    llama_backend_init();
    llama_numa_init(params.numa);

    callback_data cb_data;
    params.cb_eval = cb_get_hidden;
    params.cb_eval_user_data = &cb_data;

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;
    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    common_init_result llama_init_tgt = common_init_from_params(params);
    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    if (!model_tgt || !ctx_tgt) {
        fprintf(stderr, "Error: failed to load target model from '%s'\n", params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }
    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;

    common_init_result llama_init_dft = common_init_from_params(params);
    model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    if (!model_dft || !ctx_dft) {
        fprintf(stderr, "Error: failed to load draft model from '%s'\n", params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    // LM HEAD SHARING
    {
        struct ggml_tensor * tgt_output = llama_get_model(ctx_tgt)->output;
        if (!tgt_output) {
            fprintf(stderr, "Error: target model output tensor is NULL\n");
            return 1;
        }
        const_cast<struct llama_model *>(llama_get_model(ctx_dft))->output = tgt_output;
        auto * mem_dft_init = llama_get_memory(ctx_dft);
        llama_memory_clear(mem_dft_init, false);
        if (llama_get_model(ctx_tgt)->output_norm && !llama_get_model(ctx_dft)->output_norm) {
            const_cast<struct llama_model *>(llama_get_model(ctx_dft))->output_norm = llama_get_model(ctx_tgt)->output_norm;
        }
        fprintf(stderr, "[Spec-Bench] LM head sharing: OK\n");
    }

    // Vocab check
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    if (llama_vocab_type(vocab_tgt) != llama_vocab_type(vocab_dft)) {
        fprintf(stderr, "Error: vocab type mismatch\n");
        return 1;
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;
    const int n_draft_max = params.speculative.n_max;

    // Pre-allocate batches (reused across prompts)
    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    fprintf(stderr, "[Spec-Bench] Models loaded. Starting benchmark...\n\n");

    // ====================================================================
    // Prompt loop
    // ====================================================================
    std::vector<bench_result> results;
    token_freq_stats token_stats;

    for (size_t prompt_idx = 0; prompt_idx < prompts.size(); ++prompt_idx) {
        const auto & bp = prompts[prompt_idx];

        std::string prompt_text = apply_template(chat_template, bp.text);

        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "[%zu/%zu] id=%d category=%s\n", prompt_idx + 1, prompts.size(), bp.question_id, bp.category.c_str());
        fprintf(stderr, "  prompt: %.80s%s\n", bp.text.c_str(), bp.text.size() > 80 ? "..." : "");
        fprintf(stderr, "  --- output start ---\n");

        bench_result res = {};
        res.question_id = bp.question_id;
        res.category = bp.category;
        res.success = false;

        // ------ Reset state ------
        llama_memory_clear(mem_tgt, true);
        llama_memory_clear(mem_dft, true);

        // ------ Tokenize ------
        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt_text, true, true);
        res.n_input = (int)inp.size();

        if ((int)inp.size() > max_tokens_list_size) {
            fprintf(stderr, "  SKIP: prompt too long (%d tokens, max %d)\n", (int)inp.size(), max_tokens_list_size);
            results.push_back(res);
            continue;
        }

        // ------ Sampler ------
        struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

        // ------ Prefill ------
        const auto t_enc_start = ggml_time_us();

        llama_batch temp_batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
        int temp_n_past = 0;
        for (size_t i = 0; i + 1 < inp.size(); i++) {
            common_batch_add(temp_batch_tgt, inp[i], temp_n_past++, { 0 }, true);
        }

        cb_data.data.clear();
        llama_decode(ctx_tgt, temp_batch_tgt);
        ctx_tgt->synchronize();
        std::vector<float> sliced_data(cb_data.data.begin(), cb_data.data.end());

        cb_data.data.clear();
        llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1));
        std::vector<float> backup_data(cb_data.data.begin(), cb_data.data.end());

        cb_data.data.clear();
        llama_decode_eagle(ctx_dft, llama_batch_get_one(inp.data() + 1, (int)inp.size() - 1), sliced_data.data());

        const auto t_enc_end = ggml_time_us();
        llama_batch_free(temp_batch_tgt);

        // ------ Decode state init ------
        const int n_input = (int)inp.size();
        int n_predict = 0;
        int n_drafted = 0;
        int n_accept  = 0;
        int n_past_tgt = n_input;
        int n_past_dft = n_input - 1;
        bool has_eos = false;

        std::vector<seq_draft> drafts(n_seq_dft);
        std::vector<int> acceptance_lengths;
        std::vector<int> decoding_latencies;
        std::vector<int> verification_latencies;
        std::vector<float> T_d;

        int cur_depth = 0;
        int third_depth[4] = { 0, 1, 4, 5 };

        std::vector<std::vector<float>> scores(n_seq_dft, std::vector<float>(n_depth, 0.0f));
        std::vector<float> column_scores(n_seq_dft, 0.0f);

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
        }

        drafts[0].i_batch_tgt.resize(1);
        drafts[0].i_batch_tgt[0] = 0;

        const auto t_dec_start = ggml_time_us();
        auto verification_start = ggml_time_us();

        // ====================================================================
        // Speculative decode loop (identical to speculative-eagle.cpp)
        // ====================================================================
        while (true) {
            std::set<int> active_seqs = {};
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].active) continue;
                active_seqs.insert(s);
            }

            int i_dft  = 0;
            int s_keep = 0;
            llama_token token_id;
            std::string token_str;
            std::vector<float> temp2;
            std::vector<llama_token> recompute;

            // ---- Verification loop ----
            while (true) {
                {
                    bool accept = false;
                    if (params.sampling.temp > 0) {
                        // stochastic verification
                        common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);
                        auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                        float p_tgt = 0.0f, p_dft = 0.0f;

                        while (active_seqs.size() > 0) {
                            std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                            int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                            if (i_dft >= (int)drafts[s].tokens.size()) {
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
                                if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) { p_tgt = dist_tgt.data[i].p; break; }
                            }
                            for (size_t i = 0; i < dist_dft.size; i++) {
                                if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) { p_dft = dist_dft.data[i].p; break; }
                            }

                            if (r <= p_tgt / p_dft) {
                                s_keep = s; accept = true;
                                token_id = drafts[s].tokens[i_dft];
                                token_str = common_token_to_piece(ctx_tgt, token_id);
                                common_sampler_accept(smpl, token_id, true);
                                break;
                            } else {
                                drafts[s].active = false;
                                GGML_ASSERT(dist_tgt.sorted);
                                GGML_ASSERT(dist_dft.sorted);
                                std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) { return a.id < b.id; });
                                std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) { return a.id < b.id; });
                                float sum_probs = 0.0f;
                                for (size_t i = 0; i < dist_tgt.size; i++) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - (i < dist_dft.size ? dist_dft.data[i].p : 0.0f));
                                    sum_probs += dist_tgt.data[i].p;
                                }
                                for (size_t i = 0; i < dist_tgt.size; i++) dist_tgt.data[i].p /= sum_probs;
                                std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) { return a.p > b.p; });
                            }
                            active_seqs.erase(s);
                            for (int ii = 0; ii < n_seq_dft; ii++) {
                                if (ii == s) continue;
                                if (drafts[ii].active && drafts[ii].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                    drafts[ii].active = drafts[ii].active && accept;
                                    if (!drafts[ii].active) active_seqs.erase(ii);
                                }
                            }
                        }
                        if (!accept) {
                            std::vector<float> probs(dist_tgt.size);
                            for (size_t i = 0; i < dist_tgt.size; ++i) probs[i] = dist_tgt.data[i].p;
                            std::discrete_distribution<> dist(probs.begin(), probs.end());
                            token_id = dist_tgt.data[dist(rng)].id;
                            common_sampler_accept(smpl, token_id, true);
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                        }
                    } else {
                        // greedy verification
                        token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                        temp2.insert(temp2.end(), backup_data.begin() + (4096 * drafts[s_keep].i_batch_tgt[i_dft]),
                                                  backup_data.begin() + (4096 * (drafts[s_keep].i_batch_tgt[i_dft] + 1)));
                        recompute.push_back(token_id);
                        for (int s = 0; s < n_seq_dft; ++s) {
                            if (!drafts[s].active) continue;
                            if (i_dft < (int)drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                                s_keep = s; accept = true;
                            } else {
                                drafts[s].active = false;
                            }
                        }
                    }

                    if (llama_vocab_is_eog(vocab_tgt, token_id)) has_eos = true;
                    ++n_predict;

                    // print and accumulate generated token
                    if (!has_eos) {
                        printf("%s", token_str.c_str());
                        fflush(stdout);
                        res.output_text += token_str;
                    }

                    if (accept) {
                        ++n_accept; ++n_past_tgt; ++n_past_dft; ++i_dft;
                        token_stats.draft_accepted[token_id]++;
                        continue;
                    } else {
                        token_stats.bonus_freq[token_id]++;
                        break;
                    }
                }
            } // end verification loop

            const auto verification_end = ggml_time_us();
            verification_latencies.push_back((int)((verification_end - verification_start) / 1000));
            acceptance_lengths.push_back(i_dft + 1);

            for (int i = 0; i < n_seq_dft; i++)
                for (int j = 0; j < n_depth; j++)
                    scores[i][j] = 0.0f;

            backup_data = temp2;
            std::vector<float> temp3(backup_data.end() - std::min((size_t)4096, backup_data.size()), backup_data.end());
            int recompute_point = n_past_dft - i_dft;

            // ---- Drafting ----
            const auto drafting_start = ggml_time_us();

            bool dft_exhausted = false;

            // Recompute logic
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
                    for (size_t i = 0; i + 1 < recompute.size(); i++) {
                        common_batch_add(batch_dft, recompute[i], recompute_point + (int)i, { 0 }, false);
                    }
                    cb_data.data.clear();
                    if (llama_decode_eagle(ctx_dft, batch_dft, temp4.data()) != 0) {
                        LOG_WRN("draft model KV cache exhausted (recompute), falling back\n");
                        dft_exhausted = true;
                    }
                }

                if (!dft_exhausted) {
                    common_batch_clear(batch_dft);
                    common_batch_add(batch_dft, token_id, n_past_dft, {0}, true);
                    cb_data.data.clear();
                    if (llama_decode_eagle(ctx_dft, batch_dft, temp3.data()) != 0) {
                        LOG_WRN("draft model KV cache exhausted (single token), falling back\n");
                        dft_exhausted = true;
                    } else {
                        ++n_past_dft;
                    }
                }
            }

            if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos || dft_exhausted) {
                break;
            }

            if (drafts[0].smpl) common_sampler_free(drafts[0].smpl);
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

            // ---- Tree drafting ----
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

            for (int i = 0; i < n_draft_max; ++i) {
                batch_dft.n_tokens = 0;
                if (batch_tgt.n_tokens >= n_draft_max) break;
                if (i >= 5) break;

                // Skip logic per depth
                if (cur_depth < 2) {
                    for (int s = 0; s < n_seq_dft; ++s) drafts[s].skip = false;
                } else if (cur_depth == 2) {
                    for (int s = 0; s < n_seq_dft; ++s) {
                        int in = 0;
                        for (int ii = 0; ii < 4; ii++) { if (s == third_depth[ii]) in = 1; }
                        drafts[s].skip = (in == 0);
                    }
                } else {
                    for (int s = 0; s < n_seq_dft; ++s) {
                        drafts[s].skip = (s != 0);
                    }
                }

                std::vector<float> temp;

                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].drafting || drafts[s].skip) continue;

                    common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
                    const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);

                    std::vector<int> sa(1, s);
                    temp.insert(temp.end(), cb_data.data.begin() + (4096 * s), cb_data.data.begin() + (4096 * (s + 1)));

                    float prob = cur_p->data[0].p;
                    if (i == 0) { scores[s][i] = prob; column_scores[s] = prob; }
                    else { scores[s][i] = scores[s][i-1] * prob; column_scores[s] = scores[s][i-1] * prob; }

                    // Split logic (Draft Budget 25 tree)
                    int f_max = 4;
                    if (cur_depth == 0) {
                        f_max = 4;
                    } else if (cur_depth == 1) {
                        if      (s == 0) f_max = 3;
                        else if (s == 1) f_max = 2;
                        else if (s == 2) f_max = 2;
                        else if (s == 3) f_max = 1;
                        else f_max = 1;
                    } else if (cur_depth == 2) {
                        if      (s == 0) f_max = 3;
                        else if (s == 1) f_max = 1;
                        else if (s == 4) f_max = 2;
                        else if (s == 5) f_max = 2;
                        else f_max = 1;
                    } else if (cur_depth == 3) {
                        if (s == 0) f_max = 3;
                        else f_max = 1;
                    } else if (cur_depth == 4) {
                        f_max = 2;
                    } else {
                        f_max = 4;
                    }

                    for (int f = 1; f < f_max; ++f) {
                        if (n_seq_cur < n_seq_dft) {
                            llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                            llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);
                            temp.insert(temp.end(), cb_data.data.begin() + (4096 * s), cb_data.data.begin() + (4096 * (s + 1)));
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

                            float fp = cur_p->data[f].p;
                            if (i == 0) { scores[n_seq_cur-1][i] = fp; column_scores[n_seq_cur-1] = fp; }
                            else { scores[n_seq_cur-1][i] = scores[s][i-1] * fp; column_scores[n_seq_cur-1] = scores[s][i-1] * fp; }
                        } else break;
                    }

                    // Add tokens
                    for (int is = 0; is < (int)sa.size(); ++is) {
                        const llama_token id = cur_p->data[is].id;
                        const int ss = sa[is];
                        token_stats.draft_freq[id]++;
                        common_sampler_accept(drafts[ss].smpl, id, true);
                        drafts[ss].tokens.push_back(id);
                        drafts[ss].dists.push_back({cur_p->data, cur_p->data + cur_p->size});
                        drafts[ss].i_batch_tgt.push_back(batch_tgt.n_tokens);
                        common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { ss }, true);
                        drafts[ss].i_batch_dft = batch_dft.n_tokens;

                        if (cur_depth == 0) {
                            common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 1) {
                            int in = 0;
                            for (int ii = 0; ii < 4; ii++) { if (ss == third_depth[ii]) in = 1; }
                            if (in) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 2) {
                            if (ss == 0) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                        } else if (cur_depth == 3) {
                            if (ss == 0) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                            else drafts[ss].drafting = false;
                        } else if (cur_depth == 4) {
                            if (ss == 0) common_batch_add(batch_dft, id, n_past_cur, { ss }, true);
                            else drafts[ss].drafting = false;
                        }
                        if (batch_tgt.n_tokens > n_draft_max) drafts[ss].drafting = false;
                    }
                }

                if (batch_dft.n_tokens == 0) break;
                if (batch_tgt.n_tokens > n_draft_max) break;

                const auto dft_t0 = ggml_time_us();
                cb_data.data.clear();
                if (llama_decode_eagle(ctx_dft, batch_dft, temp.data()) != 0) {
                    LOG_WRN("draft model KV cache exhausted (tree drafting), stopping drafting\n");
                    break;
                }
                ctx_dft->synchronize();
                const auto dft_t1 = ggml_time_us();
                if (batch_dft.n_tokens == 1) T_d.push_back((dft_t1 - dft_t0) / 1000.0f);
                ++n_past_cur;
                ++n_drafted;
                cur_depth += 1;
            }
            cur_depth = 0;

            const auto drafting_end = ggml_time_us();
            decoding_latencies.push_back((int)((drafting_end - drafting_start) / 1000.0f));

            verification_start = ggml_time_us();

            // Evaluate target model on drafted tokens
            {
                llama_memory_seq_keep(mem_tgt, 0);
                for (int s = 1; s < n_seq_dft; ++s) {
                    llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
                }
                cb_data.data.clear();
                llama_decode(ctx_tgt, batch_tgt);
                ctx_tgt->synchronize();
                backup_data = cb_data.data;
                ++n_past_tgt;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].active) continue;
                drafts[s].tokens.erase(drafts[s].tokens.begin());
                drafts[s].dists.erase(drafts[s].dists.begin());
            }
        } // end speculative decode loop

        const auto t_dec_end = ggml_time_us();

        // ---- Collect per-prompt stats ----
        res.n_predict = n_predict;
        res.n_drafted = n_drafted;
        res.n_accept  = n_accept;
        res.prefill_ms  = (t_enc_end - t_enc_start) / 1000.0;
        res.decode_ms   = (t_dec_end - t_dec_start) / 1000.0;
        res.prefill_tps = n_input / (res.prefill_ms / 1000.0);
        res.decode_tps  = n_predict > 0 ? n_predict / (res.decode_ms / 1000.0) : 0;
        res.decode_lat  = n_predict > 0 ? res.decode_ms / n_predict : 0;
        res.accept_ratio = n_drafted > 0 ? 100.0 * n_accept / n_drafted : 0;

        int n_steps = (int)decoding_latencies.size();
        res.avg_accept_len = n_steps > 0 ? std::accumulate(acceptance_lengths.begin() + 1, acceptance_lengths.end(), 0.0) / n_steps : 0;
        res.avg_draft_lat = !decoding_latencies.empty() ? std::accumulate(decoding_latencies.begin(), decoding_latencies.end(), 0.0) / decoding_latencies.size() : 0;
        res.avg_verify_lat = !verification_latencies.empty() ? std::accumulate(verification_latencies.begin(), verification_latencies.end(), 0.0) / verification_latencies.size() : 0;
        res.avg_td = !T_d.empty() ? std::accumulate(T_d.begin(), T_d.end(), 0.0) / T_d.size() : 0;
        res.success = true;

        printf("\n");
        fprintf(stderr, "  --- output end ---\n");
        fprintf(stderr, "  -> %d tokens | %.2f t/s | accept_len=%.2f | accept_ratio=%.1f%%\n",
                n_predict, res.decode_tps, res.avg_accept_len, res.accept_ratio);

        results.push_back(res);

        // ---- Cleanup per-prompt resources ----
        common_sampler_free(smpl);
        for (int s = 0; s < n_seq_dft; ++s) {
            if (drafts[s].smpl) { common_sampler_free(drafts[s].smpl); drafts[s].smpl = nullptr; }
        }
    } // end prompt loop

    // ====================================================================
    // Aggregate results
    // ====================================================================
    fprintf(stderr, "\n\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "            Spec-Bench Results (%zu prompts)\n", results.size());
    fprintf(stderr, "============================================================\n");

    // Per-category aggregation
    std::map<std::string, std::vector<const bench_result *>> by_category;
    for (const auto & r : results) {
        if (r.success) by_category[r.category].push_back(&r);
    }

    auto print_group = [](const char * label, const std::vector<const bench_result *> & grp) {
        if (grp.empty()) return;
        double sum_tps = 0, sum_al = 0, sum_ar = 0, sum_lat = 0;
        int count = 0;
        for (const auto * r : grp) {
            sum_tps += r->decode_tps;
            sum_al  += r->avg_accept_len;
            sum_ar  += r->accept_ratio;
            sum_lat += r->decode_lat;
            count++;
        }
        fprintf(stderr, "  %-20s : %3d prompts | %6.2f t/s | accept_len=%5.2f | accept_ratio=%5.1f%% | lat=%6.2f ms/tok\n",
                label, count, sum_tps / count, sum_al / count, sum_ar / count, sum_lat / count);
    };

    for (const auto & [cat, grp] : by_category) {
        print_group(cat.c_str(), grp);
    }

    fprintf(stderr, "------------------------------------------------------------\n");
    std::vector<const bench_result *> all_success;
    for (const auto & r : results) { if (r.success) all_success.push_back(&r); }
    print_group("OVERALL", all_success);

    int n_skipped = 0;
    for (const auto & r : results) { if (!r.success) n_skipped++; }
    if (n_skipped > 0) {
        fprintf(stderr, "  Skipped: %d prompts\n", n_skipped);
    }
    fprintf(stderr, "============================================================\n");

    // Write CSV (metrics only)
    {
        std::string csv_path = bench_file + "_results.csv";
        std::ofstream csv(csv_path);
        if (csv.is_open()) {
            csv << "question_id,category,n_input,n_predict,decode_tps,decode_lat_ms,accept_len,accept_ratio,prefill_ms,avg_draft_ms,avg_verify_ms,avg_td_ms\n";
            for (const auto & r : results) {
                if (!r.success) continue;
                csv << r.question_id << "," << r.category << "," << r.n_input << "," << r.n_predict << ","
                    << r.decode_tps << "," << r.decode_lat << "," << r.avg_accept_len << "," << r.accept_ratio << ","
                    << r.prefill_ms << "," << r.avg_draft_lat << "," << r.avg_verify_lat << "," << r.avg_td << "\n";
            }
            fprintf(stderr, "\nMetrics saved to: %s\n", csv_path.c_str());
        }
    }

    // Write JSONL with outputs
    {
        std::string jsonl_path = bench_file + "_outputs.jsonl";
        std::ofstream ofs(jsonl_path);
        if (ofs.is_open()) {
            for (const auto & r : results) {
                if (!r.success) continue;
                // escape quotes and backslashes in output text for JSON
                std::string escaped;
                for (char c : r.output_text) {
                    if (c == '"') escaped += "\\\"";
                    else if (c == '\\') escaped += "\\\\";
                    else if (c == '\n') escaped += "\\n";
                    else if (c == '\r') escaped += "\\r";
                    else if (c == '\t') escaped += "\\t";
                    else escaped += c;
                }
                ofs << "{\"question_id\":" << r.question_id
                    << ",\"category\":\"" << r.category << "\""
                    << ",\"decode_tps\":" << r.decode_tps
                    << ",\"accept_len\":" << r.avg_accept_len
                    << ",\"output\":\"" << escaped << "\"}\n";
            }
            fprintf(stderr, "Outputs saved to: %s\n", jsonl_path.c_str());
        }
    }

    // ====================================================================
    // Token frequency stats for vocab compression
    // ====================================================================
    {
        std::unordered_map<llama_token, int64_t> all_tokens;
        for (const auto & [tid, cnt] : token_stats.draft_freq)    all_tokens[tid] += 0;
        for (const auto & [tid, cnt] : token_stats.draft_accepted) all_tokens[tid] += 0;
        for (const auto & [tid, cnt] : token_stats.bonus_freq)    all_tokens[tid] += 0;

        int64_t total_draft = 0, total_accepted = 0, total_bonus = 0;
        for (const auto & [tid, cnt] : token_stats.draft_freq)    total_draft    += cnt;
        for (const auto & [tid, cnt] : token_stats.draft_accepted) total_accepted += cnt;
        for (const auto & [tid, cnt] : token_stats.bonus_freq)    total_bonus    += cnt;

        fprintf(stderr, "\n============================================================\n");
        fprintf(stderr, "  Token Frequency Stats (for vocab compression)\n");
        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "  Unique tokens drafted  : %zu\n", token_stats.draft_freq.size());
        fprintf(stderr, "  Unique tokens accepted : %zu\n", token_stats.draft_accepted.size());
        fprintf(stderr, "  Unique bonus tokens    : %zu\n", token_stats.bonus_freq.size());
        fprintf(stderr, "  Total draft count      : %lld\n", (long long)total_draft);
        fprintf(stderr, "  Total accepted count   : %lld\n", (long long)total_accepted);
        fprintf(stderr, "  Total bonus count      : %lld\n", (long long)total_bonus);
        if (total_draft > 0) {
            fprintf(stderr, "  Overall accept rate    : %.2f%%\n", 100.0 * total_accepted / total_draft);
        }
        fprintf(stderr, "============================================================\n");

        std::string freq_path = bench_file + "_token_freq.csv";
        std::ofstream freq_csv(freq_path);
        if (freq_csv.is_open()) {
            freq_csv << "token_id,token_text,draft_count,accepted_count,rejected_count,bonus_count,accept_rate\n";

            struct token_row {
                llama_token id;
                int64_t draft, accepted, bonus;
            };
            std::vector<token_row> rows;
            for (const auto & [tid, _] : all_tokens) {
                token_row r;
                r.id       = tid;
                r.draft    = token_stats.draft_freq.count(tid)    ? token_stats.draft_freq.at(tid)    : 0;
                r.accepted = token_stats.draft_accepted.count(tid) ? token_stats.draft_accepted.at(tid) : 0;
                r.bonus    = token_stats.bonus_freq.count(tid)    ? token_stats.bonus_freq.at(tid)    : 0;
                rows.push_back(r);
            }
            std::sort(rows.begin(), rows.end(), [](const token_row & a, const token_row & b) {
                return (a.draft + a.bonus) > (b.draft + b.bonus);
            });

            for (const auto & r : rows) {
                std::string tok_text = common_token_to_piece(ctx_tgt, r.id);
                std::string escaped;
                for (char c : tok_text) {
                    if (c == '"')  escaped += "\"\"";
                    else if (c == '\n') escaped += "\\n";
                    else if (c == '\r') escaped += "\\r";
                    else if (c == '\t') escaped += "\\t";
                    else escaped += c;
                }
                int64_t rejected = r.draft > r.accepted ? r.draft - r.accepted : 0;
                double accept_rate = r.draft > 0 ? 100.0 * r.accepted / r.draft : 0.0;
                freq_csv << r.id << ",\"" << escaped << "\"," << r.draft << ","
                         << r.accepted << "," << rejected << "," << r.bonus << ","
                         << accept_rate << "\n";
            }
            fprintf(stderr, "\nToken frequency stats saved to: %s\n", freq_path.c_str());
        }
    }

    // ====================================================================
    // Cleanup
    // ====================================================================
    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);
    llama_backend_free();

    return 0;
}
