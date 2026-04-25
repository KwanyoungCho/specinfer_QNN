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
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <functional>
#include <limits>
#include <utility>
#include <unordered_map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

#define SPEC_BENCH_BACKEND_LABEL "Spec-Bench"
#define SPEC_BENCH_RESULTS_SUFFIX "_results"

#define n_depth 5
#define expand_k 4
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
    int    sample_index;
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
    double draft_len;
    double avg_accept_len;
    double avg_accepted_prefix_len;
    double avg_step_output_len;
    double accept_ratio;
    double avg_draft_lat;
    double avg_verify_lat;
    double avg_td;
    bool   success;
    std::string error_message;
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

using length_hist = std::map<int, int64_t>;
using token_total_map = std::map<llama_token, int64_t>;
using token_pos_count_map = std::map<llama_token, std::map<int, int64_t>>;

struct token_position_stats {
    token_pos_count_map verified_pos_count;
    token_pos_count_map accepted_pos_count;
    token_pos_count_map bonus_pos_count;
    token_pos_count_map proposed_pos_count;

    length_hist accepted_prefix_hist;
    length_hist step_output_hist;

    token_total_map verified_total;
    token_total_map accepted_total;
    token_total_map bonus_total;
    token_total_map proposed_total;
};

struct shortlist_coverage_row {
    int64_t verified_total = 0;
    int64_t verified_hit   = 0;
    int64_t verified_miss  = 0;
    int64_t accepted_total = 0;
    int64_t accepted_hit   = 0;
    int64_t accepted_miss  = 0;
    int64_t bonus_total    = 0;
    int64_t bonus_hit      = 0;
    int64_t bonus_miss     = 0;
};

struct shortlist_coverage_stats {
    std::map<int, shortlist_coverage_row> by_position;
};

struct analysis_stats {
    token_position_stats overall;
    std::map<std::string, token_position_stats> by_category;

    shortlist_coverage_stats coverage_overall;
    std::map<std::string, shortlist_coverage_stats> coverage_by_category;

    token_total_map target_generated_freq;
    std::map<std::string, token_total_map> target_generated_freq_by_category;
};

struct shortlist_config {
    std::string global_path;
    std::string category_dir;
    bool enabled = false;
    bool save_trace = false;

    std::set<llama_token> global_tokens;
    std::map<std::string, std::set<llama_token>> category_tokens;
};

// ============================================================
// JSONL prompt loader
// ============================================================

struct bench_prompt {
    int question_id;
    std::string category;
    std::string text;
};

struct bench_summary {
    std::string label;
    int total_prompts = 0;
    int successful_prompts = 0;
    int skipped_prompts = 0;
    int64_t total_input_tokens = 0;
    int64_t total_decode_tokens = 0;
    int64_t total_drafted_tokens = 0;
    int64_t total_accepted_tokens = 0;
    double avg_prefill_ms = 0.0;
    double avg_decode_ms = 0.0;
    double avg_prefill_tps = 0.0;
    double avg_decode_tps = 0.0;
    double avg_decode_lat = 0.0;
    double avg_draft_len = 0.0;
    double avg_accept_len = 0.0;
    double avg_accepted_prefix_len = 0.0;
    double avg_step_output_len = 0.0;
    double avg_accept_ratio = 0.0;
    double avg_draft_lat = 0.0;
    double avg_verify_lat = 0.0;
    double avg_td = 0.0;
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

static std::string csv_escape(const std::string & text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char c : text) {
        if (c == '"') {
            escaped += "\"\"";
        } else if (c == '\n') {
            escaped += "\\n";
        } else if (c == '\r') {
            escaped += "\\r";
        } else if (c == '\t') {
            escaped += "\\t";
        } else {
            escaped += c;
        }
    }
    return escaped;
}

static std::string json_escape(const std::string & text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char c : text) {
        if (c == '"') escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else escaped += c;
    }
    return escaped;
}

static std::string normalize_category_key(const std::string & category) {
    return category.empty() ? "UNCATEGORIZED" : category;
}

static std::string normalize_shortlist_category_name(const std::filesystem::path & path) {
    std::string stem = path.stem().string();
    static const std::string suffix = "_topK";
    if (stem.size() > suffix.size() &&
        stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
        stem.resize(stem.size() - suffix.size());
    }
    return normalize_category_key(stem);
}

static int64_t lookup_total_count(const token_total_map & counts, llama_token token_id) {
    const auto it = counts.find(token_id);
    return it == counts.end() ? 0 : it->second;
}

static int64_t lookup_pos_count(
    const token_pos_count_map & counts,
    llama_token token_id,
    int position) {
    const auto tok_it = counts.find(token_id);
    if (tok_it == counts.end()) {
        return 0;
    }
    const auto pos_it = tok_it->second.find(position);
    return pos_it == tok_it->second.end() ? 0 : pos_it->second;
}

static void add_total_count(token_total_map & counts, llama_token token_id, int64_t delta = 1) {
    counts[token_id] += delta;
}

static void add_pos_count(
    token_pos_count_map & counts,
    llama_token token_id,
    int position,
    int64_t delta = 1) {
    counts[token_id][position] += delta;
}

static void add_hist_count(length_hist & hist, int value, int64_t delta = 1) {
    hist[value] += delta;
}

static token_position_stats & get_category_stats(analysis_stats & stats, const std::string & category) {
    return stats.by_category[normalize_category_key(category)];
}

static shortlist_coverage_stats & get_category_coverage(
    analysis_stats & stats,
    const std::string & category) {
    return stats.coverage_by_category[normalize_category_key(category)];
}

static token_total_map & get_category_target_generated_freq(
    analysis_stats & stats,
    const std::string & category) {
    return stats.target_generated_freq_by_category[normalize_category_key(category)];
}

static void record_verified_step_stats(
    token_position_stats & stats,
    const std::vector<llama_token> & verified_tokens,
    int accepted_prefix_len) {
    const int step_output_len = (int) verified_tokens.size();
    add_hist_count(stats.accepted_prefix_hist, accepted_prefix_len);
    add_hist_count(stats.step_output_hist, step_output_len);

    for (int i = 0; i < step_output_len; ++i) {
        const int position = i + 1;
        const llama_token token_id = verified_tokens[i];

        add_pos_count(stats.verified_pos_count, token_id, position);
        add_total_count(stats.verified_total, token_id);

        if (i < accepted_prefix_len) {
            add_pos_count(stats.accepted_pos_count, token_id, position);
            add_total_count(stats.accepted_total, token_id);
        } else {
            add_pos_count(stats.bonus_pos_count, token_id, position);
            add_total_count(stats.bonus_total, token_id);
        }
    }
}

static void record_verified_step_stats(
    analysis_stats & stats,
    const std::string & category,
    const std::vector<llama_token> & verified_tokens,
    int accepted_prefix_len) {
    record_verified_step_stats(stats.overall, verified_tokens, accepted_prefix_len);
    record_verified_step_stats(get_category_stats(stats, category), verified_tokens, accepted_prefix_len);
}

static void record_proposed_token_stats(
    token_position_stats & stats,
    llama_token token_id,
    int position) {
    add_pos_count(stats.proposed_pos_count, token_id, position);
    add_total_count(stats.proposed_total, token_id);
}

static void record_proposed_token_stats(
    analysis_stats & stats,
    const std::string & category,
    llama_token token_id,
    int position) {
    record_proposed_token_stats(stats.overall, token_id, position);
    record_proposed_token_stats(get_category_stats(stats, category), token_id, position);
}

static void record_target_generated_token(
    analysis_stats & stats,
    const std::string & category,
    llama_token token_id) {
    add_total_count(stats.target_generated_freq, token_id);
    add_total_count(get_category_target_generated_freq(stats, category), token_id);
}

static bool load_shortlist_tokens_from_file(
    const std::string & path,
    std::set<llama_token> & tokens) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }
        for (char & c : line) {
            if (c == ',' || c == ';' || c == '\t') {
                c = ' ';
            }
        }

        std::stringstream ss(line);
        long long token_id = 0;
        while (ss >> token_id) {
            tokens.insert((llama_token) token_id);
        }
    }

    return true;
}

static bool load_shortlist_config(shortlist_config & cfg) {
    cfg.enabled = false;
    cfg.global_tokens.clear();
    cfg.category_tokens.clear();

    if (!cfg.global_path.empty()) {
        if (!load_shortlist_tokens_from_file(cfg.global_path, cfg.global_tokens)) {
            fprintf(stderr, "Error: failed to load analysis shortlist: %s\n", cfg.global_path.c_str());
            return false;
        }
        cfg.enabled = true;
        fprintf(stderr, "[Spec-Bench] Loaded global analysis shortlist: %zu tokens from %s\n",
                cfg.global_tokens.size(), cfg.global_path.c_str());
    }

    if (!cfg.category_dir.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(cfg.category_dir, ec)) {
            fprintf(stderr, "Error: analysis shortlist directory does not exist: %s\n", cfg.category_dir.c_str());
            return false;
        }

        for (const auto & entry : std::filesystem::directory_iterator(cfg.category_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            std::set<llama_token> tokens;
            const std::string path = entry.path().string();
            if (!load_shortlist_tokens_from_file(path, tokens)) {
                fprintf(stderr, "Warning: failed to load category shortlist: %s\n", path.c_str());
                continue;
            }
            if (tokens.empty()) {
                continue;
            }

            const std::string category = normalize_shortlist_category_name(entry.path());
            cfg.category_tokens[category] = std::move(tokens);
            cfg.enabled = true;
        }

        fprintf(stderr, "[Spec-Bench] Loaded %zu category shortlists from %s\n",
                cfg.category_tokens.size(), cfg.category_dir.c_str());
    }

    return true;
}

static const std::set<llama_token> * resolve_shortlist_tokens(
    const shortlist_config & cfg,
    const std::string & category) {
    const std::string key = normalize_category_key(category);
    const auto cat_it = cfg.category_tokens.find(key);
    if (cat_it != cfg.category_tokens.end()) {
        return &cat_it->second;
    }
    if (!cfg.global_tokens.empty()) {
        return &cfg.global_tokens;
    }
    return nullptr;
}

static void update_hit_miss(
    int64_t & total,
    int64_t & hit,
    int64_t & miss,
    bool covered) {
    ++total;
    if (covered) {
        ++hit;
    } else {
        ++miss;
    }
}

static void record_coverage_step(
    shortlist_coverage_stats & stats,
    const std::vector<llama_token> & verified_tokens,
    int accepted_prefix_len,
    const std::set<llama_token> & shortlist) {
    for (int i = 0; i < (int) verified_tokens.size(); ++i) {
        const int position = i + 1;
        const bool covered = shortlist.find(verified_tokens[i]) != shortlist.end();
        auto & row = stats.by_position[position];

        update_hit_miss(row.verified_total, row.verified_hit, row.verified_miss, covered);
        if (i < accepted_prefix_len) {
            update_hit_miss(row.accepted_total, row.accepted_hit, row.accepted_miss, covered);
        } else {
            update_hit_miss(row.bonus_total, row.bonus_hit, row.bonus_miss, covered);
        }
    }
}

static void record_coverage_step(
    analysis_stats & stats,
    const std::string & category,
    const std::vector<llama_token> & verified_tokens,
    int accepted_prefix_len,
    const shortlist_config & cfg) {
    const auto * shortlist = resolve_shortlist_tokens(cfg, category);
    if (!shortlist) {
        return;
    }

    record_coverage_step(stats.coverage_overall, verified_tokens, accepted_prefix_len, *shortlist);
    record_coverage_step(get_category_coverage(stats, category), verified_tokens, accepted_prefix_len, *shortlist);
}

static std::vector<bool> compute_shortlist_hit_flags(
    const std::vector<llama_token> & verified_tokens,
    const shortlist_config & cfg,
    const std::string & category) {
    std::vector<bool> hits(verified_tokens.size(), false);
    const auto * shortlist = resolve_shortlist_tokens(cfg, category);
    if (!shortlist) {
        return hits;
    }

    for (size_t i = 0; i < verified_tokens.size(); ++i) {
        hits[i] = shortlist->find(verified_tokens[i]) != shortlist->end();
    }
    return hits;
}

static void write_accept_hist_rows(
    std::ofstream & csv,
    const std::string & category,
    const char * kind,
    const length_hist & hist) {
    for (const auto & [length, count] : hist) {
        csv << "\"" << csv_escape(kind) << "\","
            << length << ","
            << count << ","
            << "\"" << csv_escape(category) << "\"\n";
    }
}

static void write_accept_hist_csv(
    const std::string & results_dir,
    const analysis_stats & stats) {
    const std::string path = (std::filesystem::path(results_dir) / "accept_hist.csv").string();
    std::ofstream csv(path);
    if (!csv.is_open()) {
        fprintf(stderr, "Warning: failed to write accept histogram CSV: %s\n", path.c_str());
        return;
    }

    csv << "kind,length,count,category\n";
    write_accept_hist_rows(csv, "OVERALL", "accepted_prefix", stats.overall.accepted_prefix_hist);
    write_accept_hist_rows(csv, "OVERALL", "step_output",     stats.overall.step_output_hist);
    for (const auto & [category, cat_stats] : stats.by_category) {
        write_accept_hist_rows(csv, category, "accepted_prefix", cat_stats.accepted_prefix_hist);
        write_accept_hist_rows(csv, category, "step_output",     cat_stats.step_output_hist);
    }

    fprintf(stderr, "Accept hist saved to: %s\n", path.c_str());
}

static void write_token_pos_rows(
    std::ofstream & csv,
    llama_context * ctx_tgt,
    const std::string & category,
    const token_position_stats & stats) {
    std::set<llama_token> token_ids;
    for (const auto & [token_id, _] : stats.verified_pos_count) token_ids.insert(token_id);
    for (const auto & [token_id, _] : stats.accepted_pos_count) token_ids.insert(token_id);
    for (const auto & [token_id, _] : stats.bonus_pos_count)    token_ids.insert(token_id);
    for (const auto & [token_id, _] : stats.proposed_pos_count) token_ids.insert(token_id);

    for (const auto & token_id : token_ids) {
        std::set<int> positions;
        const auto gather_positions = [&](const token_pos_count_map & map) {
            const auto tok_it = map.find(token_id);
            if (tok_it == map.end()) {
                return;
            }
            for (const auto & [position, _] : tok_it->second) {
                positions.insert(position);
            }
        };

        gather_positions(stats.verified_pos_count);
        gather_positions(stats.accepted_pos_count);
        gather_positions(stats.bonus_pos_count);
        gather_positions(stats.proposed_pos_count);

        const std::string token_text = common_token_to_piece(ctx_tgt, token_id);
        for (const int position : positions) {
            csv << "\"" << csv_escape(category) << "\","
                << token_id << ","
                << "\"" << csv_escape(token_text) << "\","
                << position << ","
                << lookup_pos_count(stats.verified_pos_count, token_id, position) << ","
                << lookup_pos_count(stats.accepted_pos_count, token_id, position) << ","
                << lookup_pos_count(stats.bonus_pos_count, token_id, position) << ","
                << lookup_pos_count(stats.proposed_pos_count, token_id, position) << "\n";
        }
    }
}

static void write_token_pos_stats_csv(
    const std::string & results_dir,
    llama_context * ctx_tgt,
    const analysis_stats & stats) {
    const std::string path = (std::filesystem::path(results_dir) / "token_pos_stats.csv").string();
    std::ofstream csv(path);
    if (!csv.is_open()) {
        fprintf(stderr, "Warning: failed to write token position stats CSV: %s\n", path.c_str());
        return;
    }

    csv << "category,token_id,token_text,position,verified_count,accepted_count,bonus_count,proposed_count\n";
    write_token_pos_rows(csv, ctx_tgt, "OVERALL", stats.overall);
    for (const auto & [category, cat_stats] : stats.by_category) {
        write_token_pos_rows(csv, ctx_tgt, category, cat_stats);
    }

    fprintf(stderr, "Token position stats saved to: %s\n", path.c_str());
}

static void write_shortlist_coverage_rows(
    std::ofstream & csv,
    const std::string & category,
    const shortlist_coverage_stats & stats) {
    for (const auto & [position, row] : stats.by_position) {
        const double miss_rate_verified = row.verified_total > 0 ? (double) row.verified_miss / row.verified_total : 0.0;
        const double miss_rate_accepted = row.accepted_total > 0 ? (double) row.accepted_miss / row.accepted_total : 0.0;
        const double miss_rate_bonus    = row.bonus_total    > 0 ? (double) row.bonus_miss    / row.bonus_total    : 0.0;

        csv << "\"" << csv_escape(category) << "\","
            << position << ","
            << row.verified_total << ","
            << row.verified_hit << ","
            << row.verified_miss << ","
            << row.accepted_total << ","
            << row.accepted_hit << ","
            << row.accepted_miss << ","
            << row.bonus_total << ","
            << row.bonus_hit << ","
            << row.bonus_miss << ","
            << miss_rate_verified << ","
            << miss_rate_accepted << ","
            << miss_rate_bonus << "\n";
    }
}

static void write_shortlist_coverage_csv(
    const std::string & results_dir,
    const analysis_stats & stats,
    const shortlist_config & cfg) {
    if (!cfg.enabled) {
        return;
    }

    const std::string path = (std::filesystem::path(results_dir) / "shortlist_coverage.csv").string();
    std::ofstream csv(path);
    if (!csv.is_open()) {
        fprintf(stderr, "Warning: failed to write shortlist coverage CSV: %s\n", path.c_str());
        return;
    }

    csv << "category,position,verified_total,verified_hit,verified_miss,"
        << "accepted_total,accepted_hit,accepted_miss,"
        << "bonus_total,bonus_hit,bonus_miss,"
        << "miss_rate_verified,miss_rate_accepted,miss_rate_bonus\n";

    write_shortlist_coverage_rows(csv, "OVERALL", stats.coverage_overall);
    for (const auto & [category, cat_stats] : stats.coverage_by_category) {
        write_shortlist_coverage_rows(csv, category, cat_stats);
    }

    fprintf(stderr, "Shortlist coverage saved to: %s\n", path.c_str());
}

static void write_target_generated_freq_rows(
    std::ofstream & csv,
    llama_context * ctx_tgt,
    const std::string & category,
    const token_total_map & counts) {
    for (const auto & [token_id, count] : counts) {
        csv << "\"" << csv_escape(category) << "\","
            << token_id << ","
            << "\"" << csv_escape(common_token_to_piece(ctx_tgt, token_id)) << "\","
            << count << "\n";
    }
}

static void write_target_generated_freq_csv(
    const std::string & results_dir,
    llama_context * ctx_tgt,
    const analysis_stats & stats) {
    const std::string path = (std::filesystem::path(results_dir) / "target_generated_freq.csv").string();
    std::ofstream csv(path);
    if (!csv.is_open()) {
        fprintf(stderr, "Warning: failed to write target-generated frequency CSV: %s\n", path.c_str());
        return;
    }

    csv << "category,token_id,token_text,count\n";
    write_target_generated_freq_rows(csv, ctx_tgt, "OVERALL", stats.target_generated_freq);
    for (const auto & [category, counts] : stats.target_generated_freq_by_category) {
        write_target_generated_freq_rows(csv, ctx_tgt, category, counts);
    }

    fprintf(stderr, "Target-generated frequency saved to: %s\n", path.c_str());
}

static bench_summary summarize_results(
    const std::string & label,
    const std::vector<const bench_result *> & group) {
    bench_summary summary;
    summary.label = label;
    summary.total_prompts = (int) group.size();

    for (const auto * result : group) {
        if (!result) {
            continue;
        }

        if (!result->success) {
            summary.skipped_prompts++;
            continue;
        }

        summary.successful_prompts++;
        summary.total_input_tokens += result->n_input;
        summary.total_decode_tokens += result->n_predict;
        summary.total_drafted_tokens += result->n_drafted;
        summary.total_accepted_tokens += result->n_accept;
        summary.avg_prefill_ms += result->prefill_ms;
        summary.avg_decode_ms += result->decode_ms;
        summary.avg_prefill_tps += result->prefill_tps;
        summary.avg_decode_tps += result->decode_tps;
        summary.avg_decode_lat += result->decode_lat;
        summary.avg_draft_len += result->draft_len;
        summary.avg_accept_len += result->avg_accept_len;
        summary.avg_accepted_prefix_len += result->avg_accepted_prefix_len;
        summary.avg_step_output_len += result->avg_step_output_len;
        summary.avg_accept_ratio += result->accept_ratio;
        summary.avg_draft_lat += result->avg_draft_lat;
        summary.avg_verify_lat += result->avg_verify_lat;
        summary.avg_td += result->avg_td;
    }

    summary.skipped_prompts = summary.total_prompts - summary.successful_prompts;

    if (summary.successful_prompts > 0) {
        const double denom = (double) summary.successful_prompts;
        summary.avg_prefill_ms /= denom;
        summary.avg_decode_ms /= denom;
        summary.avg_prefill_tps /= denom;
        summary.avg_decode_tps /= denom;
        summary.avg_decode_lat /= denom;
        summary.avg_draft_len /= denom;
        summary.avg_accept_len /= denom;
        summary.avg_accepted_prefix_len /= denom;
        summary.avg_step_output_len /= denom;
        summary.avg_accept_ratio /= denom;
        summary.avg_draft_lat /= denom;
        summary.avg_verify_lat /= denom;
        summary.avg_td /= denom;
    }

    return summary;
}

static void print_summary_row(FILE * stream, const bench_summary & summary) {
    if (summary.successful_prompts == 0) {
        fprintf(stream, "  %-20s :   0/%3d ok | no successful prompts\n",
                summary.label.c_str(), summary.total_prompts);
        return;
    }

    fprintf(stream,
            "  %-20s : %3d/%3d ok | %6.2f t/s | draft_len=%5.2f | accept_len=%5.2f | accept_ratio=%5.1f%% | lat=%6.2f ms/tok\n",
            summary.label.c_str(),
            summary.successful_prompts,
            summary.total_prompts,
            summary.avg_decode_tps,
            summary.avg_draft_len,
            summary.avg_accept_len,
            summary.avg_accept_ratio,
            summary.avg_decode_lat);
}

static void write_summary_csv(
    const std::string & results_dir,
    const std::vector<bench_summary> & summaries) {
    const std::string summary_path = (std::filesystem::path(results_dir) / "summary.csv").string();
    std::ofstream summary_csv(summary_path);
    if (!summary_csv.is_open()) {
        fprintf(stderr, "Warning: failed to write summary CSV: %s\n", summary_path.c_str());
        return;
    }

    summary_csv
        << "group,total_prompts,successful_prompts,skipped_prompts,total_input_tokens,total_decode_tokens,"
        << "total_drafted_tokens,total_accepted_tokens,avg_prefill_ms,avg_decode_ms,avg_prefill_tps,"
        << "avg_decode_tps,avg_decode_lat_ms,avg_draft_len,avg_accept_len,avg_accepted_prefix_len,"
        << "avg_step_output_len,avg_accept_ratio,"
        << "avg_draft_ms,avg_verify_ms,avg_td_ms\n";

    for (const auto & summary : summaries) {
        summary_csv
            << "\"" << csv_escape(summary.label) << "\","
            << summary.total_prompts << ","
            << summary.successful_prompts << ","
            << summary.skipped_prompts << ","
            << summary.total_input_tokens << ","
            << summary.total_decode_tokens << ","
            << summary.total_drafted_tokens << ","
            << summary.total_accepted_tokens << ","
            << summary.avg_prefill_ms << ","
            << summary.avg_decode_ms << ","
            << summary.avg_prefill_tps << ","
            << summary.avg_decode_tps << ","
            << summary.avg_decode_lat << ","
            << summary.avg_draft_len << ","
            << summary.avg_accept_len << ","
            << summary.avg_accepted_prefix_len << ","
            << summary.avg_step_output_len << ","
            << summary.avg_accept_ratio << ","
            << summary.avg_draft_lat << ","
            << summary.avg_verify_lat << ","
            << summary.avg_td << "\n";
    }

    fprintf(stderr, "Summary saved to: %s\n", summary_path.c_str());
}

static std::string default_results_dir(const std::string & bench_file) {
    namespace fs = std::filesystem;
    fs::path p(bench_file);
    fs::path parent = p.parent_path();
    std::string stem = p.stem().empty() ? p.filename().string() : p.stem().string();
    return (parent / (stem + SPEC_BENCH_RESULTS_SUFFIX)).string();
}

static std::string make_prompt_output_path(const std::string & results_dir, int sample_index) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "output_%05d.txt", sample_index);
    return (std::filesystem::path(results_dir) / buf).string();
}

static void append_hidden_state_slice(
    std::vector<float> & dst,
    const std::vector<float> & src,
    int hidden_dim,
    int index) {
    if (hidden_dim <= 0 || index < 0) {
        return;
    }

    const size_t begin = (size_t) hidden_dim * (size_t) index;
    const size_t end   = begin + (size_t) hidden_dim;
    if (end > src.size()) {
        return;
    }

    dst.insert(dst.end(), src.begin() + begin, src.begin() + end);
}

static std::vector<float> take_last_hidden_states(
    const std::vector<float> & src,
    int hidden_dim,
    int n_tokens) {
    if (hidden_dim <= 0 || n_tokens <= 0) {
        return {};
    }

    const size_t n_values = std::min(src.size(), (size_t) hidden_dim * (size_t) n_tokens);
    if (n_values == 0) {
        return {};
    }

    return std::vector<float>(src.end() - n_values, src.end());
}

static std::vector<float> take_first_hidden_states(
    const std::vector<float> & src,
    int hidden_dim,
    int n_tokens) {
    if (hidden_dim <= 0 || n_tokens <= 0) {
        return {};
    }

    const size_t n_values = std::min(src.size(), (size_t) hidden_dim * (size_t) n_tokens);
    return std::vector<float>(src.begin(), src.begin() + n_values);
}

static void write_prompt_result_file(
    const std::string & path,
    const bench_result & res,
    const std::string & raw_prompt) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        fprintf(stderr, "Warning: failed to write prompt result file: %s\n", path.c_str());
        return;
    }

    ofs << "============================================================\n";
    ofs << "  " << SPEC_BENCH_BACKEND_LABEL << " Sample Result\n";
    ofs << "============================================================\n";
    ofs << "Sample index       : " << res.sample_index << "\n";
    ofs << "Question ID        : " << res.question_id << "\n";
    ofs << "Category           : " << res.category << "\n";
    ofs << "Status             : " << (res.success ? "success" : "skipped") << "\n";
    if (!res.error_message.empty()) {
        ofs << "Skip reason        : " << res.error_message << "\n";
    }
    ofs << "------------------------------------------------------------\n";
    ofs << "Prompt:\n";
    ofs << raw_prompt << "\n";
    ofs << "------------------------------------------------------------\n";
    ofs << "Output:\n";
    ofs << res.output_text << "\n";
    ofs << "------------------------------------------------------------\n";

    if (!res.success) {
        return;
    }

    ofs << "  Prefill           : " << res.n_input << " tokens | " << res.prefill_ms << " ms | " << res.prefill_tps << " t/s\n";
    ofs << "  Decode            : " << res.n_predict << " tokens | " << res.decode_ms << " ms | " << res.decode_tps << " t/s\n";
    ofs << "  Decode latency    :              | " << res.decode_lat << " ms/tok\n";
    ofs << "------------------------------------------------------------\n";
    ofs << "  Draft length      : " << res.draft_len << "\n";
    ofs << "  Avg accept length (legacy step_output) : " << res.avg_accept_len << "\n";
    ofs << "  Avg accepted prefix len               : " << res.avg_accepted_prefix_len << "\n";
    ofs << "  Avg step output len                   : " << res.avg_step_output_len << "\n";
    ofs << "  Accept ratio      : " << res.accept_ratio << "%\n";
    ofs << "------------------------------------------------------------\n";
    ofs << "  Avg draft phase   : " << res.avg_draft_lat << " ms\n";
    ofs << "  Avg verification  : " << res.avg_verify_lat << " ms\n";
    ofs << "  Avg T_d (1-tok dft) : " << res.avg_td << " ms\n";
}

// ============================================================
// Selector training data collection
// ============================================================

struct selector_data_config {
    bool collect = false;
    std::string data_dir;
    std::string source = "generated";
    int lookahead_depth = 5;
    int top_k = 2048;
    int64_t max_samples = -1;
    bool save_hidden_fp16 = true;
    bool save_logits_fp16 = true;
};

struct selector_topk_row {
    std::vector<int32_t> ids;
    std::vector<float> logits;
};

struct selector_cached_row {
    std::vector<float> hidden;
    std::vector<int32_t> top_ids;
    std::vector<float> top_logits;
};

static bool write_float_record(
    std::ofstream & ofs,
    const float * data,
    size_t count,
    bool fp16,
    std::vector<ggml_fp16_t> & fp16_scratch) {
    if (!ofs.is_open() || !data) {
        return false;
    }

    if (fp16) {
        fp16_scratch.resize(count);
        for (size_t i = 0; i < count; ++i) {
            fp16_scratch[i] = ggml_fp32_to_fp16(data[i]);
        }
        ofs.write(reinterpret_cast<const char *>(fp16_scratch.data()),
                  (std::streamsize) (fp16_scratch.size() * sizeof(ggml_fp16_t)));
    } else {
        ofs.write(reinterpret_cast<const char *>(data),
                  (std::streamsize) (count * sizeof(float)));
    }

    return ofs.good();
}

static void topk_logits(
    const float * logits,
    int vocab_size,
    int k,
    std::vector<int32_t> & ids,
    std::vector<float> & values) {
    ids.assign(k, -1);
    values.assign(k, -std::numeric_limits<float>::infinity());

    if (!logits || vocab_size <= 0 || k <= 0) {
        return;
    }

    const int k_eff = std::min(k, vocab_size);
    std::vector<std::pair<float, int32_t>> heap;
    heap.reserve(k_eff);
    const auto cmp = std::greater<std::pair<float, int32_t>>();

    for (int32_t token_id = 0; token_id < vocab_size; ++token_id) {
        const std::pair<float, int32_t> cur(logits[token_id], token_id);
        if ((int) heap.size() < k_eff) {
            heap.push_back(cur);
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (cur > heap.front()) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = cur;
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }

    std::sort(heap.begin(), heap.end(),
        [](const auto & a, const auto & b) {
            if (a.first != b.first) {
                return a.first > b.first;
            }
            return a.second < b.second;
        });

    for (int i = 0; i < k_eff; ++i) {
        values[i] = heap[i].first;
        ids[i] = heap[i].second;
    }
}

class SelectorDataWriter {
public:
    bool open(const selector_data_config & cfg) {
        cfg_ = cfg;
        namespace fs = std::filesystem;

        std::error_code ec;
        fs::create_directories(cfg_.data_dir, ec);
        if (ec) {
            fprintf(stderr, "Error: failed to create selector data directory %s: %s\n",
                    cfg_.data_dir.c_str(), ec.message().c_str());
            return false;
        }

        const fs::path dir(cfg_.data_dir);
        hidden_path_     = (dir / (cfg_.save_hidden_fp16 ? "hidden.fp16.bin" : "hidden.f32.bin")).string();
        top_ids_path_    = (dir / "top_ids.i32.bin").string();
        top_logits_path_ = (dir / (cfg_.save_logits_fp16 ? "top_logits.fp16.bin" : "top_logits.f32.bin")).string();
        gold_ids_path_   = (dir / "gold_ids.i32.bin").string();
        meta_path_       = (dir / "meta.json").string();

        hidden_.open(hidden_path_, std::ios::binary);
        top_ids_.open(top_ids_path_, std::ios::binary);
        top_logits_.open(top_logits_path_, std::ios::binary);
        gold_ids_.open(gold_ids_path_, std::ios::binary);

        if (!hidden_.is_open() || !top_ids_.is_open() ||
            !top_logits_.is_open() || !gold_ids_.is_open()) {
            fprintf(stderr, "Error: failed to open selector data output files under %s\n",
                    cfg_.data_dir.c_str());
            return false;
        }

        return true;
    }

    bool write_sample(
        const float * hidden,
        int hidden_dim,
        int vocab_size,
        const std::vector<int32_t> & top_ids,
        const std::vector<float> & top_logits,
        const std::vector<int32_t> & gold_ids) {
        if (hidden_dim <= 0 || vocab_size <= 0) {
            fprintf(stderr, "Error: invalid selector sample dimensions hidden_dim=%d vocab_size=%d\n",
                    hidden_dim, vocab_size);
            return false;
        }

        if (hidden_dim_ == 0) {
            hidden_dim_ = hidden_dim;
        } else if (hidden_dim_ != hidden_dim) {
            fprintf(stderr, "Error: selector hidden_dim changed from %d to %d\n",
                    hidden_dim_, hidden_dim);
            return false;
        }

        if (vocab_size_ == 0) {
            vocab_size_ = vocab_size;
        } else if (vocab_size_ != vocab_size) {
            fprintf(stderr, "Error: selector vocab_size changed from %d to %d\n",
                    vocab_size_, vocab_size);
            return false;
        }

        const size_t dk = (size_t) cfg_.lookahead_depth * (size_t) cfg_.top_k;
        if (top_ids.size() != dk || top_logits.size() != dk ||
            gold_ids.size() != (size_t) cfg_.lookahead_depth) {
            fprintf(stderr, "Error: invalid selector sample payload sizes\n");
            return false;
        }

        if (!write_float_record(hidden_, hidden, (size_t) hidden_dim_, cfg_.save_hidden_fp16, fp16_scratch_)) {
            fprintf(stderr, "Error: failed to write selector hidden record\n");
            return false;
        }

        top_ids_.write(reinterpret_cast<const char *>(top_ids.data()),
                       (std::streamsize) (top_ids.size() * sizeof(int32_t)));
        if (!top_ids_.good()) {
            fprintf(stderr, "Error: failed to write selector top_ids record\n");
            return false;
        }

        if (!write_float_record(top_logits_, top_logits.data(), top_logits.size(),
                                cfg_.save_logits_fp16, fp16_scratch_)) {
            fprintf(stderr, "Error: failed to write selector top_logits record\n");
            return false;
        }

        gold_ids_.write(reinterpret_cast<const char *>(gold_ids.data()),
                        (std::streamsize) (gold_ids.size() * sizeof(int32_t)));
        if (!gold_ids_.good()) {
            fprintf(stderr, "Error: failed to write selector gold_ids record\n");
            return false;
        }

        ++num_samples_;
        return true;
    }

    void close() {
        hidden_.close();
        top_ids_.close();
        top_logits_.close();
        gold_ids_.close();
    }

    bool write_meta(
        const std::string & bench_file,
        const std::string & dataset_type,
        const std::string & chat_template,
        int prompts_processed,
        int64_t token_positions_processed) const {
        std::ofstream ofs(meta_path_);
        if (!ofs.is_open()) {
            fprintf(stderr, "Error: failed to write selector meta file: %s\n", meta_path_.c_str());
            return false;
        }

        ofs << "{\n"
            << "  \"num_samples\": " << num_samples_ << ",\n"
            << "  \"hidden_dim\": " << hidden_dim_ << ",\n"
            << "  \"vocab_size\": " << vocab_size_ << ",\n"
            << "  \"lookahead_depth\": " << cfg_.lookahead_depth << ",\n"
            << "  \"top_k\": " << cfg_.top_k << ",\n"
            << "  \"hidden_dtype\": \"" << (cfg_.save_hidden_fp16 ? "fp16" : "f32") << "\",\n"
            << "  \"top_logits_dtype\": \"" << (cfg_.save_logits_fp16 ? "fp16" : "f32") << "\",\n"
            << "  \"top_ids_dtype\": \"i32\",\n"
            << "  \"gold_ids_dtype\": \"i32\",\n"
            << "  \"bench_file\": \"" << json_escape(bench_file) << "\",\n"
            << "  \"dataset_type\": \"" << json_escape(dataset_type) << "\",\n"
            << "  \"chat_template\": \"" << json_escape(chat_template) << "\",\n"
            << "  \"alignment_description\": \"hidden[t] and logits[t] predict tokens[t+1]. top_ids[t,0] is from logits[t]. gold_ids[t,0] is tokens[t+1].\",\n"
            << "  \"prompts_processed\": " << prompts_processed << ",\n"
            << "  \"token_positions_processed\": " << token_positions_processed << "\n"
            << "}\n";

        return ofs.good();
    }

    int64_t num_samples() const { return num_samples_; }
    int hidden_dim() const { return hidden_dim_; }
    int vocab_size() const { return vocab_size_; }
    void set_vocab_size(int vocab_size) { if (vocab_size_ == 0) vocab_size_ = vocab_size; }

private:
    selector_data_config cfg_;
    int64_t num_samples_ = 0;
    int hidden_dim_ = 0;
    int vocab_size_ = 0;

    std::string hidden_path_;
    std::string top_ids_path_;
    std::string top_logits_path_;
    std::string gold_ids_path_;
    std::string meta_path_;

    std::ofstream hidden_;
    std::ofstream top_ids_;
    std::ofstream top_logits_;
    std::ofstream gold_ids_;
    std::vector<ggml_fp16_t> fp16_scratch_;
};

static int collect_selector_prompt_data(
    const selector_data_config & cfg,
    const std::vector<bench_prompt> & prompts,
    size_t idx_start,
    size_t idx_end,
    const std::string & bench_file,
    const std::string & dataset_type,
    const std::string & chat_template,
    llama_context * ctx_tgt,
    llama_model * model_tgt,
    callback_data & cb_data) {
    if (!ctx_tgt || !model_tgt) {
        fprintf(stderr, "Error: selector collector received null model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    int vocab_size = (int) llama_model_n_vocab_out(model_tgt);
    if (vocab_size <= 0 && vocab) {
        vocab_size = llama_vocab_n_tokens(vocab);
    }
    if (vocab_size <= 0) {
        fprintf(stderr, "Error: failed to infer selector vocab size from target model\n");
        return 1;
    }
    if (cfg.top_k > vocab_size) {
        fprintf(stderr, "Error: --selector-top-k (%d) exceeds vocab size (%d)\n",
                cfg.top_k, vocab_size);
        return 1;
    }

    SelectorDataWriter writer;
    if (!writer.open(cfg)) {
        return 1;
    }
    writer.set_vocab_size(vocab_size);

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    const int max_context_size = (int) llama_n_ctx(ctx_tgt);
    const int max_batch_size = (int) llama_n_batch(ctx_tgt);
    int64_t token_positions_processed = 0;
    int prompts_processed = 0;
    int prompts_with_samples = 0;

    fprintf(stderr,
            "[%s] Selector data collection enabled: dir=%s D=%d K=%d max_samples=%" PRId64 "\n",
            SPEC_BENCH_BACKEND_LABEL,
            cfg.data_dir.c_str(),
            cfg.lookahead_depth,
            cfg.top_k,
            cfg.max_samples);

    for (size_t prompt_idx = idx_start; prompt_idx < idx_end; ++prompt_idx) {
        if (cfg.max_samples >= 0 && writer.num_samples() >= cfg.max_samples) {
            break;
        }

        const auto & bp = prompts[prompt_idx];
        const std::string prompt_text = apply_template(chat_template, bp.text);
        std::vector<llama_token> tokens = common_tokenize(ctx_tgt, prompt_text, true, true);
        const int n_tokens = (int) tokens.size();

        ++prompts_processed;

        if (n_tokens <= cfg.lookahead_depth) {
            if (prompts_processed % 10 == 0) {
                fprintf(stderr,
                        "[%s] selector progress: prompts=%d token_positions=%" PRId64 " samples=%" PRId64 "\n",
                        SPEC_BENCH_BACKEND_LABEL,
                        prompts_processed,
                        token_positions_processed,
                        writer.num_samples());
            }
            continue;
        }

        if (n_tokens > max_context_size) {
            fprintf(stderr,
                    "[%s] selector skip prompt %zu: %d tokens exceed context limit (%d)\n",
                    SPEC_BENCH_BACKEND_LABEL,
                    prompt_idx + 1,
                    n_tokens,
                    max_context_size);
            continue;
        }

        llama_memory_clear(mem_tgt, true);
        cb_data.data.clear();

        const int rows_needed = n_tokens - 1;
        std::vector<selector_topk_row> topk_by_row(rows_needed);
        bool ok = true;
        int n_past = 0;
        llama_batch batch = llama_batch_init(max_batch_size, 0, 1);

        for (int chunk_start = 0; chunk_start < rows_needed; chunk_start += max_batch_size) {
            const int chunk_size = std::min(max_batch_size, rows_needed - chunk_start);
            common_batch_clear(batch);
            for (int j = 0; j < chunk_size; ++j) {
                common_batch_add(batch, tokens[chunk_start + j], n_past++, { 0 }, true);
            }

            if (llama_decode(ctx_tgt, batch) != 0) {
                fprintf(stderr,
                        "[%s] selector skip prompt %zu: target decode failed at token %d\n",
                        SPEC_BENCH_BACKEND_LABEL,
                        prompt_idx + 1,
                        chunk_start);
                ok = false;
                break;
            }
            ctx_tgt->synchronize();

            for (int j = 0; j < chunk_size; ++j) {
                const float * logits = llama_get_logits_ith(ctx_tgt, j);
                if (!logits) {
                    fprintf(stderr,
                            "[%s] selector skip prompt %zu: logits row %d unavailable\n",
                            SPEC_BENCH_BACKEND_LABEL,
                            prompt_idx + 1,
                            chunk_start + j);
                    ok = false;
                    break;
                }
                topk_logits(logits, vocab_size, cfg.top_k,
                            topk_by_row[chunk_start + j].ids,
                            topk_by_row[chunk_start + j].logits);
            }

            if (!ok) {
                break;
            }
        }

        llama_batch_free(batch);

        if (!ok) {
            continue;
        }

        const auto & hiddens = cb_data.data;
        if (hiddens.empty() || hiddens.size() % (size_t) rows_needed != 0) {
            fprintf(stderr,
                    "[%s] selector skip prompt %zu: result_norm hidden states unavailable (floats=%zu rows=%d)\n",
                    SPEC_BENCH_BACKEND_LABEL,
                    prompt_idx + 1,
                    hiddens.size(),
                    rows_needed);
            continue;
        }

        const int hidden_dim = (int) (hiddens.size() / (size_t) rows_needed);
        if (hidden_dim <= 0) {
            fprintf(stderr,
                    "[%s] selector skip prompt %zu: invalid hidden_dim=%d\n",
                    SPEC_BENCH_BACKEND_LABEL,
                    prompt_idx + 1,
                    hidden_dim);
            continue;
        }

        const int n_valid_positions = n_tokens - cfg.lookahead_depth;
        bool prompt_wrote_sample = false;
        std::vector<int32_t> sample_top_ids;
        std::vector<float> sample_top_logits;
        std::vector<int32_t> sample_gold_ids;
        sample_top_ids.reserve((size_t) cfg.lookahead_depth * (size_t) cfg.top_k);
        sample_top_logits.reserve((size_t) cfg.lookahead_depth * (size_t) cfg.top_k);
        sample_gold_ids.reserve(cfg.lookahead_depth);

        for (int t = 0; t < n_valid_positions; ++t) {
            if (cfg.max_samples >= 0 && writer.num_samples() >= cfg.max_samples) {
                break;
            }

            sample_top_ids.clear();
            sample_top_logits.clear();
            sample_gold_ids.clear();

            for (int d = 0; d < cfg.lookahead_depth; ++d) {
                const auto & row = topk_by_row[t + d];
                sample_top_ids.insert(sample_top_ids.end(), row.ids.begin(), row.ids.end());
                sample_top_logits.insert(sample_top_logits.end(), row.logits.begin(), row.logits.end());
                sample_gold_ids.push_back((int32_t) tokens[t + d + 1]);
            }

            const float * hidden = hiddens.data() + (size_t) t * (size_t) hidden_dim;
            if (!writer.write_sample(hidden, hidden_dim, vocab_size,
                                     sample_top_ids, sample_top_logits, sample_gold_ids)) {
                writer.close();
                return 1;
            }

            ++token_positions_processed;
            prompt_wrote_sample = true;
        }

        if (prompt_wrote_sample) {
            ++prompts_with_samples;
        }

        if (prompts_processed % 10 == 0 ||
            prompt_idx + 1 == idx_end ||
            (cfg.max_samples >= 0 && writer.num_samples() >= cfg.max_samples)) {
            fprintf(stderr,
                    "[%s] selector progress: prompts=%d token_positions=%" PRId64 " samples=%" PRId64 "\n",
                    SPEC_BENCH_BACKEND_LABEL,
                    prompts_processed,
                    token_positions_processed,
                    writer.num_samples());
        }
    }

    writer.close();

    if (!writer.write_meta(bench_file, dataset_type, chat_template,
                           prompts_processed, token_positions_processed)) {
        return 1;
    }

    fprintf(stderr,
            "[%s] selector collection complete: samples=%" PRId64 " prompts=%d prompts_with_samples=%d hidden_dim=%d vocab_size=%d\n",
            SPEC_BENCH_BACKEND_LABEL,
            writer.num_samples(),
            prompts_processed,
            prompts_with_samples,
            writer.hidden_dim(),
            writer.vocab_size());

    return 0;
}

static bool selector_try_emit_generated_samples(
    const selector_data_config & cfg,
    SelectorDataWriter & writer,
    std::deque<selector_cached_row> & rows,
    std::deque<llama_token> & tokens,
    int hidden_dim,
    int vocab_size,
    int64_t & token_positions_processed) {
    while ((int) rows.size() >= cfg.lookahead_depth &&
           (int) tokens.size() >= cfg.lookahead_depth + 1) {
        if (cfg.max_samples >= 0 && writer.num_samples() >= cfg.max_samples) {
            return true;
        }

        std::vector<int32_t> sample_top_ids;
        std::vector<float> sample_top_logits;
        std::vector<int32_t> sample_gold_ids;
        sample_top_ids.reserve((size_t) cfg.lookahead_depth * (size_t) cfg.top_k);
        sample_top_logits.reserve((size_t) cfg.lookahead_depth * (size_t) cfg.top_k);
        sample_gold_ids.reserve(cfg.lookahead_depth);

        for (int d = 0; d < cfg.lookahead_depth; ++d) {
            sample_top_ids.insert(sample_top_ids.end(), rows[d].top_ids.begin(), rows[d].top_ids.end());
            sample_top_logits.insert(sample_top_logits.end(), rows[d].top_logits.begin(), rows[d].top_logits.end());
            sample_gold_ids.push_back((int32_t) tokens[d + 1]);
        }

        if (!writer.write_sample(rows.front().hidden.data(), hidden_dim, vocab_size,
                                 sample_top_ids, sample_top_logits, sample_gold_ids)) {
            return false;
        }

        ++token_positions_processed;
        rows.pop_front();
        tokens.pop_front();
    }

    return true;
}

static int collect_selector_generated_data(
    const selector_data_config & cfg,
    const std::vector<bench_prompt> & prompts,
    size_t idx_start,
    size_t idx_end,
    const std::string & bench_file,
    const std::string & dataset_type,
    const std::string & chat_template,
    const common_params & params,
    llama_context * ctx_tgt,
    llama_model * model_tgt,
    callback_data & cb_data) {
    if (!ctx_tgt || !model_tgt) {
        fprintf(stderr, "Error: selector collector received null model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    int vocab_size = (int) llama_model_n_vocab_out(model_tgt);
    if (vocab_size <= 0 && vocab) {
        vocab_size = llama_vocab_n_tokens(vocab);
    }
    if (vocab_size <= 0) {
        fprintf(stderr, "Error: failed to infer selector vocab size from target model\n");
        return 1;
    }
    if (cfg.top_k > vocab_size) {
        fprintf(stderr, "Error: --selector-top-k (%d) exceeds vocab size (%d)\n",
                cfg.top_k, vocab_size);
        return 1;
    }

    SelectorDataWriter writer;
    if (!writer.open(cfg)) {
        return 1;
    }
    writer.set_vocab_size(vocab_size);

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    const int max_context_size = (int) llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;
    const int n_batch_tgt = (int) llama_n_batch(ctx_tgt);
    const int hidden_dim = llama_model_n_embd(model_tgt);
    if (hidden_dim <= 0) {
        fprintf(stderr, "Error: failed to infer selector hidden_dim from target model\n");
        return 1;
    }

    int64_t token_positions_processed = 0;
    int prompts_processed = 0;
    int prompts_with_samples = 0;

    fprintf(stderr,
            "[%s] Selector generated-output collection enabled: dir=%s D=%d K=%d max_samples=%" PRId64 "\n",
            SPEC_BENCH_BACKEND_LABEL,
            cfg.data_dir.c_str(),
            cfg.lookahead_depth,
            cfg.top_k,
            cfg.max_samples);

    for (size_t prompt_idx = idx_start; prompt_idx < idx_end; ++prompt_idx) {
        if (cfg.max_samples >= 0 && writer.num_samples() >= cfg.max_samples) {
            break;
        }

        const auto & bp = prompts[prompt_idx];
        const std::string prompt_text = apply_template(chat_template, bp.text);
        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt_text, true, true);

        ++prompts_processed;

        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "[Selector %zu/%zu] id=%d category=%s\n",
                prompt_idx + 1, prompts.size(), bp.question_id, bp.category.c_str());
        fprintf(stderr, "  prompt: %.80s%s\n", bp.text.c_str(), bp.text.size() > 80 ? "..." : "");
        fprintf(stderr, "  --- output start ---\n");

        if ((int) inp.size() > max_tokens_list_size) {
            fprintf(stderr, "\n  SKIP: prompt too long (%d tokens, max %d)\n",
                    (int) inp.size(), max_tokens_list_size);
            continue;
        }

        llama_memory_clear(mem_tgt, true);
        cb_data.data.clear();

        struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);
        llama_batch batch = llama_batch_init(n_batch_tgt, 0, 1);

        bool ok = true;
        int n_past = 0;

        for (int chunk_start = 0; chunk_start < (int) inp.size(); chunk_start += n_batch_tgt) {
            const int chunk_size = std::min(n_batch_tgt, (int) inp.size() - chunk_start);
            common_batch_clear(batch);
            for (int j = 0; j < chunk_size; ++j) {
                const bool logits = (chunk_start + j == (int) inp.size() - 1);
                common_batch_add(batch, inp[chunk_start + j], n_past++, { 0 }, logits);
            }

            if (llama_decode(ctx_tgt, batch) != 0) {
                fprintf(stderr, "\n  SKIP: target prompt prefill failed at chunk start=%d\n", chunk_start);
                ok = false;
                break;
            }
        }

        if (ok) {
            ctx_tgt->synchronize();
        }

        selector_cached_row prompt_boundary_row;
        bool has_prompt_boundary_row = false;
        if (ok) {
            if (cb_data.data.size() < (size_t) hidden_dim) {
                fprintf(stderr,
                        "\n  SKIP: prompt-final result_norm hidden state unavailable (floats=%zu, hidden_dim=%d)\n",
                        cb_data.data.size(), hidden_dim);
                ok = false;
            } else {
                const float * prompt_logits = llama_get_logits_ith(ctx_tgt, -1);
                if (!prompt_logits) {
                    fprintf(stderr, "\n  SKIP: prompt-final logits unavailable\n");
                    ok = false;
                } else {
                    const float * last_prompt_hidden = cb_data.data.data() + cb_data.data.size() - (size_t) hidden_dim;
                    prompt_boundary_row.hidden.assign(last_prompt_hidden, last_prompt_hidden + hidden_dim);
                    topk_logits(prompt_logits, vocab_size, cfg.top_k,
                                prompt_boundary_row.top_ids,
                                prompt_boundary_row.top_logits);
                    has_prompt_boundary_row = true;
                }
            }
        }

        cb_data.data.clear();

        llama_token cur_token = LLAMA_TOKEN_NULL;
        bool has_eos = false;
        if (ok) {
            cur_token = common_sampler_sample(smpl, ctx_tgt, -1);
            common_sampler_accept(smpl, cur_token, true);
            has_eos = llama_vocab_is_eog(vocab, cur_token);
        }

        std::deque<selector_cached_row> pending_rows;
        std::deque<llama_token> token_window;
        int n_generated = 0;
        bool prompt_wrote_sample = false;

        if (ok && has_prompt_boundary_row && !has_eos) {
            pending_rows.push_back(std::move(prompt_boundary_row));
            token_window.push_back(inp.back());
            token_window.push_back(cur_token);
        }

        while (ok && !has_eos && (params.n_predict < 0 || n_generated < params.n_predict)) {
            const std::string token_str = common_token_to_piece(ctx_tgt, cur_token);
            printf("%s", token_str.c_str());
            fflush(stdout);
            ++n_generated;

            cb_data.data.clear();
            llama_token decode_token = cur_token;
            if (llama_decode(ctx_tgt, llama_batch_get_one(&decode_token, 1)) != 0) {
                fprintf(stderr, "\n  SKIP: target decode failed after %d generated tokens\n", n_generated);
                ok = false;
                break;
            }
            ctx_tgt->synchronize();

            if (cb_data.data.size() < (size_t) hidden_dim) {
                fprintf(stderr,
                        "\n  SKIP: result_norm hidden state unavailable after %d generated tokens (floats=%zu, hidden_dim=%d)\n",
                        n_generated, cb_data.data.size(), hidden_dim);
                ok = false;
                break;
            }

            const float * logits = llama_get_logits_ith(ctx_tgt, -1);
            if (!logits) {
                fprintf(stderr,
                        "\n  SKIP: logits unavailable after %d generated tokens\n",
                        n_generated);
                ok = false;
                break;
            }

            selector_cached_row row;
            row.hidden.assign(cb_data.data.begin(), cb_data.data.begin() + hidden_dim);
            topk_logits(logits, vocab_size, cfg.top_k, row.top_ids, row.top_logits);
            pending_rows.push_back(std::move(row));

            llama_token next_token = common_sampler_sample(smpl, ctx_tgt, -1);
            common_sampler_accept(smpl, next_token, true);
            token_window.push_back(next_token);

            const int64_t before = writer.num_samples();
            if (!selector_try_emit_generated_samples(cfg, writer, pending_rows, token_window,
                                                     hidden_dim, vocab_size,
                                                     token_positions_processed)) {
                ok = false;
                break;
            }
            if (writer.num_samples() > before) {
                prompt_wrote_sample = true;
            }

            if (cfg.max_samples >= 0 && writer.num_samples() >= cfg.max_samples) {
                break;
            }

            if (llama_vocab_is_eog(vocab, next_token)) {
                has_eos = true;
                break;
            }

            cur_token = next_token;
        }

        printf("\n");
        fprintf(stderr, "  --- output end ---\n");

        common_sampler_free(smpl);
        llama_batch_free(batch);

        if (prompt_wrote_sample) {
            ++prompts_with_samples;
        }

        if (!ok) {
            continue;
        }

        if (prompts_processed % 10 == 0 ||
            prompt_idx + 1 == idx_end ||
            (cfg.max_samples >= 0 && writer.num_samples() >= cfg.max_samples)) {
            fprintf(stderr,
                    "[%s] selector progress: prompts=%d generated_tokens=%d token_positions=%" PRId64 " samples=%" PRId64 "\n",
                    SPEC_BENCH_BACKEND_LABEL,
                    prompts_processed,
                    n_generated,
                    token_positions_processed,
                    writer.num_samples());
        }
    }

    writer.close();

    if (!writer.write_meta(bench_file, dataset_type, chat_template,
                           prompts_processed, token_positions_processed)) {
        return 1;
    }

    fprintf(stderr,
            "[%s] selector collection complete: samples=%" PRId64 " prompts=%d prompts_with_samples=%d hidden_dim=%d vocab_size=%d\n",
            SPEC_BENCH_BACKEND_LABEL,
            writer.num_samples(),
            prompts_processed,
            prompts_with_samples,
            writer.hidden_dim(),
            writer.vocab_size());

    return 0;
}

static void write_json_token_array(std::ostream & os, const std::vector<llama_token> & tokens) {
    os << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << tokens[i];
    }
    os << "]";
}

static void write_json_bool_array(std::ostream & os, const std::vector<bool> & values) {
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << (values[i] ? "true" : "false");
    }
    os << "]";
}

static int run_target_generate_calibration(
    const std::vector<bench_prompt> & prompts,
    size_t idx_start,
    size_t idx_end,
    const std::string & chat_template,
    const std::string & results_dir,
    const common_params & params,
    llama_model * model_tgt,
    llama_context * ctx_tgt,
    analysis_stats & analysis) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    auto * mem_tgt = llama_get_memory(ctx_tgt);
    const int max_context_size = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;
    const int n_batch_tgt = (int) llama_n_batch(ctx_tgt);

    fprintf(stderr, "[Spec-Bench] Calibration mode: target-generate\n");

    for (size_t prompt_idx = idx_start; prompt_idx < idx_end; ++prompt_idx) {
        const auto & bp = prompts[prompt_idx];
        const std::string prompt_text = apply_template(chat_template, bp.text);

        fprintf(stderr, "[Calibration %zu/%zu] id=%d category=%s\n",
                prompt_idx + 1, prompts.size(), bp.question_id, bp.category.c_str());

        llama_memory_clear(mem_tgt, true);

        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt_text, true, true);
        if ((int) inp.size() > max_tokens_list_size) {
            fprintf(stderr, "  SKIP: prompt too long (%d tokens, max %d)\n",
                    (int) inp.size(), max_tokens_list_size);
            continue;
        }

        struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);
        llama_batch batch = llama_batch_init(n_batch_tgt, 0, 1);

        bool ok = true;
        int n_past = 0;
        for (int chunk_start = 0; chunk_start < (int) inp.size(); chunk_start += n_batch_tgt) {
            const int chunk_size = std::min(n_batch_tgt, (int) inp.size() - chunk_start);
            common_batch_clear(batch);
            for (int j = 0; j < chunk_size; ++j) {
                const bool logits = (chunk_start + j == (int) inp.size() - 1);
                common_batch_add(batch, inp[chunk_start + j], n_past++, { 0 }, logits);
            }
            if (llama_decode(ctx_tgt, batch) != 0) {
                fprintf(stderr, "  SKIP: target calibration prefill failed\n");
                ok = false;
                break;
            }
        }

        if (ok) {
            ctx_tgt->synchronize();
        }

        int n_generated = 0;
        bool has_eos = false;
        std::string output_text;

        while (ok && !has_eos && (params.n_predict < 0 || n_generated < params.n_predict)) {
            llama_token token_id = common_sampler_sample(smpl, ctx_tgt, -1);
            common_sampler_accept(smpl, token_id, true);

            if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                has_eos = true;
                break;
            }

            record_target_generated_token(analysis, bp.category, token_id);
            output_text += common_token_to_piece(ctx_tgt, token_id);
            ++n_generated;

            if (llama_decode(ctx_tgt, llama_batch_get_one(&token_id, 1)) != 0) {
                fprintf(stderr, "  SKIP: target calibration decode failed\n");
                ok = false;
                break;
            }
            ctx_tgt->synchronize();
        }

        common_sampler_free(smpl);
        llama_batch_free(batch);

        if (!ok) {
            continue;
        }

        const std::string output_path = make_prompt_output_path(results_dir, (int) prompt_idx + 1);
        std::ofstream ofs(output_path);
        if (ofs.is_open()) {
            ofs << "============================================================\n";
            ofs << "  " << SPEC_BENCH_BACKEND_LABEL << " Calibration Result\n";
            ofs << "============================================================\n";
            ofs << "Sample index       : " << ((int) prompt_idx + 1) << "\n";
            ofs << "Question ID        : " << bp.question_id << "\n";
            ofs << "Category           : " << bp.category << "\n";
            ofs << "Generated tokens   : " << n_generated << "\n";
            ofs << "------------------------------------------------------------\n";
            ofs << "Prompt:\n" << bp.text << "\n";
            ofs << "------------------------------------------------------------\n";
            ofs << "Output:\n" << output_text << "\n";
        }
    }

    write_target_generated_freq_csv(results_dir, ctx_tgt, analysis);
    return 0;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char ** argv) {

    // ------ Extract custom args before common_params_parse ------
    std::string bench_file;
    std::string results_dir;
    std::string chat_template = "llama3";  // default
    std::string dataset_type  = "auto";    // auto, specbench, sharegpt
    std::string calibration_mode = "none";
    bool collect_vocab_stats = false;
    shortlist_config shortlist_cfg;
    selector_data_config selector_cfg;
    int bench_start = 0;
    int bench_count = -1;
    std::vector<char *> filtered_argv;
    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bench-file" && i + 1 < argc) {
            bench_file = argv[++i];
        } else if (arg == "--chat-template" && i + 1 < argc) {
            chat_template = argv[++i];
        } else if (arg == "--no-chat-template") {
            chat_template = "none";
        } else if (arg == "--dataset-type" && i + 1 < argc) {
            dataset_type = argv[++i];
        } else if (arg == "--bench-start" && i + 1 < argc) {
            bench_start = std::atoi(argv[++i]);
        } else if (arg == "--bench-count" && i + 1 < argc) {
            bench_count = std::atoi(argv[++i]);
        } else if (arg == "--results-dir" && i + 1 < argc) {
            results_dir = argv[++i];
        } else if (arg == "--analysis-shortlist" && i + 1 < argc) {
            shortlist_cfg.global_path = argv[++i];
        } else if (arg == "--analysis-shortlist-dir" && i + 1 < argc) {
            shortlist_cfg.category_dir = argv[++i];
        } else if (arg == "--save-trace") {
            shortlist_cfg.save_trace = true;
        } else if (arg == "--collect-vocab-stats") {
            collect_vocab_stats = true;
        } else if (arg == "--calibration-mode" && i + 1 < argc) {
            calibration_mode = argv[++i];
        } else if (arg == "--collect-selector-data") {
            selector_cfg.collect = true;
        } else if (arg == "--selector-data-dir" && i + 1 < argc) {
            selector_cfg.data_dir = argv[++i];
        } else if (arg == "--selector-source" && i + 1 < argc) {
            selector_cfg.source = argv[++i];
        } else if (arg == "--selector-lookahead-depth" && i + 1 < argc) {
            selector_cfg.lookahead_depth = std::atoi(argv[++i]);
        } else if (arg == "--selector-top-k" && i + 1 < argc) {
            selector_cfg.top_k = std::atoi(argv[++i]);
        } else if (arg == "--selector-max-samples" && i + 1 < argc) {
            selector_cfg.max_samples = std::atoll(argv[++i]);
        } else if (arg == "--selector-save-hidden-fp16") {
            selector_cfg.save_hidden_fp16 = true;
        } else if (arg == "--selector-save-logits-fp16") {
            selector_cfg.save_logits_fp16 = true;
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
        fprintf(stderr, "  --results-dir DIR        Directory for per-prompt outputs and aggregate files\n");
        fprintf(stderr, "  --analysis-shortlist F   Global shortlist token ids for analysis-only coverage\n");
        fprintf(stderr, "  --analysis-shortlist-dir D  Category shortlist directory (category filename stem)\n");
        fprintf(stderr, "  --collect-vocab-stats    Enable raw acceptance/prefix/proposal stat collection\n");
        fprintf(stderr, "  --save-trace             Save per-step verification trace to step_trace.jsonl\n");
        fprintf(stderr, "  --calibration-mode MODE  none (default) or target-generate\n");
        fprintf(stderr, "  --collect-selector-data  Collect dynamic vocab selector training data\n");
        fprintf(stderr, "  --selector-data-dir DIR  Output directory for selector data\n");
        fprintf(stderr, "  --selector-source MODE   generated (default) or prompt\n");
        fprintf(stderr, "  --selector-lookahead-depth N  Future target distributions per hidden (default: 5)\n");
        fprintf(stderr, "  --selector-top-k N       Top target logits/token ids per future step (default: 2048)\n");
        fprintf(stderr, "  --selector-max-samples N Optional sample limit (-1 = no limit)\n");
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

    if (calibration_mode != "none" && calibration_mode != "target-generate") {
        fprintf(stderr, "Error: unknown calibration mode '%s'. Use: none, target-generate\n", calibration_mode.c_str());
        return 1;
    }

    if (selector_cfg.collect) {
        if (selector_cfg.data_dir.empty()) {
            fprintf(stderr, "Error: --selector-data-dir is required with --collect-selector-data\n");
            return 1;
        }
        if (selector_cfg.lookahead_depth <= 0) {
            fprintf(stderr, "Error: --selector-lookahead-depth must be > 0\n");
            return 1;
        }
        if (selector_cfg.top_k <= 0) {
            fprintf(stderr, "Error: --selector-top-k must be > 0\n");
            return 1;
        }
        if (selector_cfg.max_samples < -1) {
            fprintf(stderr, "Error: --selector-max-samples must be -1 or >= 0\n");
            return 1;
        }
        if (selector_cfg.source != "generated" && selector_cfg.source != "prompt") {
            fprintf(stderr, "Error: unknown selector source '%s'. Use: generated, prompt\n", selector_cfg.source.c_str());
            return 1;
        }
        if (calibration_mode != "none") {
            fprintf(stderr, "Error: --collect-selector-data cannot be combined with --calibration-mode\n");
            return 1;
        }
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

    if (!selector_cfg.collect && calibration_mode == "none" && params.speculative.model.path.empty()) {
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
    if (!selector_cfg.collect) {
        if (results_dir.empty()) {
            results_dir = default_results_dir(bench_file);
        }

        std::error_code ec;
        std::filesystem::create_directories(results_dir, ec);
        if (ec) {
            fprintf(stderr, "Error: failed to create results directory %s: %s\n",
                    results_dir.c_str(), ec.message().c_str());
            return 1;
        }

        if (!load_shortlist_config(shortlist_cfg)) {
            return 1;
        }

        fprintf(stderr, "[Spec-Bench] Per-prompt results will be saved under %s\n", results_dir.c_str());
    } else {
        fprintf(stderr, "[Spec-Bench] Selector data will be saved under %s\n", selector_cfg.data_dir.c_str());
    }

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

    analysis_stats analysis;
    const size_t idx_start = std::max(0, bench_start);
    const size_t idx_end   = bench_count < 0 ? prompts.size() : std::min(prompts.size(), (size_t) (bench_start + bench_count));

    if (selector_cfg.collect) {
        const int rc = selector_cfg.source == "prompt" ?
            collect_selector_prompt_data(
                selector_cfg,
                prompts,
                idx_start,
                idx_end,
                bench_file,
                dataset_type,
                chat_template,
                ctx_tgt,
                model_tgt,
                cb_data) :
            collect_selector_generated_data(
                selector_cfg,
                prompts,
                idx_start,
                idx_end,
                bench_file,
                dataset_type,
                chat_template,
                params,
                ctx_tgt,
                model_tgt,
                cb_data);
        llama_backend_free();
        return rc;
    }

    if (calibration_mode == "target-generate") {
        const int rc = run_target_generate_calibration(
            prompts,
            idx_start,
            idx_end,
            chat_template,
            results_dir,
            params,
            model_tgt,
            ctx_tgt,
            analysis);
        llama_backend_free();
        return rc;
    }

    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }
    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;

    // draft model context must be at least as large as target model context
    params.n_ctx = llama_n_ctx(ctx_tgt);

    common_init_result llama_init_dft = common_init_from_params(params);
    model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    if (!model_dft || !ctx_dft) {
        fprintf(stderr, "Error: failed to load draft model from '%s'\n", params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    // Vocab check
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    if (llama_vocab_type(vocab_tgt) != llama_vocab_type(vocab_dft)) {
        fprintf(stderr, "Error: vocab type mismatch\n");
        return 1;
    }

    const auto & dft_vocab_map = model_dft->vocab_map;
    const bool has_vocab_trim = !dft_vocab_map.empty();

    // LM head sharing is only safe when the draft logits already use the target vocab layout.
    if (!has_vocab_trim) {
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
    } else {
        fprintf(stderr, "[Spec-Bench] LM head sharing: disabled because draft vocab is trimmed\n");
    }

    if (has_vocab_trim) {
        fprintf(stderr,
                "[Spec-Bench] Draft vocab trimming active: %zu entries in vocab_map (output_vocab_size=%u)\n",
                dft_vocab_map.size(), model_dft->hparams.n_vocab_output);
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)) {
        fprintf(stderr, "Error: draft model special tokens must match target model\n");
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = std::abs(n_vocab_tgt - n_vocab_dft);

        if (!has_vocab_trim && vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            fprintf(stderr,
                    "Error: target vocab size %d does not closely match draft vocab size %d (diff=%d, max=%d)\n",
                    n_vocab_tgt, n_vocab_dft, vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        if (!has_vocab_trim) {
            for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
                const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
                const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
                if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                    fprintf(stderr,
                            "Error: token %d differs between target and draft vocabularies ('%s' vs '%s')\n",
                            i,
                            common_token_to_piece(ctx_tgt, i).c_str(),
                            common_token_to_piece(ctx_dft, i).c_str());
                    return 1;
                }
            }
        }
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
    std::ofstream step_trace_ofs;
    if (shortlist_cfg.save_trace) {
        const std::string trace_path = (std::filesystem::path(results_dir) / "step_trace.jsonl").string();
        step_trace_ofs.open(trace_path);
        if (!step_trace_ofs.is_open()) {
            fprintf(stderr, "Warning: failed to open step trace file: %s\n", trace_path.c_str());
        } else {
            fprintf(stderr, "Step trace will be saved to: %s\n", trace_path.c_str());
        }
    }

    auto persist_result = [&](const bench_result & res, const bench_prompt & bp) {
        write_prompt_result_file(make_prompt_output_path(results_dir, res.sample_index), res, bp.text);
    };

    for (size_t prompt_idx = idx_start; prompt_idx < idx_end; ++prompt_idx) {
        const auto & bp = prompts[prompt_idx];

        std::string prompt_text = apply_template(chat_template, bp.text);

        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "[%zu/%zu] id=%d category=%s\n", prompt_idx + 1, prompts.size(), bp.question_id, bp.category.c_str());
        fprintf(stderr, "  prompt: %.80s%s\n", bp.text.c_str(), bp.text.size() > 80 ? "..." : "");
        fprintf(stderr, "  --- output start ---\n");

        bench_result res = {};
        res.sample_index = (int) prompt_idx + 1;
        res.question_id = bp.question_id;
        res.category = bp.category;
        res.success = false;

        // ------ Reset state ------
        llama_memory_clear(mem_tgt, true);
        llama_memory_clear(mem_dft, true);

        // ------ Tokenize ------
        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt_text, true, true);
        res.n_input = (int)inp.size();
        const int n_input = (int) inp.size();

        if (n_input > max_tokens_list_size) {
            fprintf(stderr, "  SKIP: prompt too long (%d tokens, max %d)\n", (int)inp.size(), max_tokens_list_size);
            res.error_message = "prompt too long";
            results.push_back(res);
            persist_result(res, bp);
            continue;
        }

        // ------ Sampler ------
        struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

        // ------ Prefill (chunked to respect n_batch limit) ------
        const auto t_enc_start = ggml_time_us();

        const int n_batch_dft = (int)llama_n_batch(ctx_dft);
        int hidden_dim = llama_model_n_embd(model_tgt);
        if (hidden_dim <= 0) {
            hidden_dim = llama_model_n_embd(model_dft);
        }

        bool tgt_prefill_ok = true;
        std::vector<float> sliced_data;
        std::vector<float> backup_data;

        const int n_batch_tgt = (int)llama_n_batch(ctx_tgt);
        const int n_prefill   = n_input - 1;
        llama_batch temp_batch_tgt = llama_batch_init(n_batch_tgt, 0, 1);
        cb_data.data.clear();
        int temp_n_past = 0;

        for (int chunk_start = 0; chunk_start < n_prefill; chunk_start += n_batch_tgt) {
            const int chunk_size = std::min(n_batch_tgt, n_prefill - chunk_start);
            common_batch_clear(temp_batch_tgt);
            for (int j = 0; j < chunk_size; j++) {
                common_batch_add(temp_batch_tgt, inp[chunk_start + j], temp_n_past++, { 0 }, true);
            }
            if (llama_decode(ctx_tgt, temp_batch_tgt) != 0) {
                fprintf(stderr, "  SKIP: target model prefill failed at chunk start=%d\n", chunk_start);
                tgt_prefill_ok = false;
                break;
            }
        }

        if (tgt_prefill_ok) {
            ctx_tgt->synchronize();
            sliced_data.assign(cb_data.data.begin(), cb_data.data.end());
            cb_data.data.clear();
            if (llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1)) != 0) {
                fprintf(stderr, "  SKIP: target model last-token prefill failed\n");
                tgt_prefill_ok = false;
            } else {
                backup_data.assign(cb_data.data.begin(), cb_data.data.end());
            }
        }

        if (!tgt_prefill_ok) {
            common_sampler_free(smpl);
            res.error_message = "target prefill failed";
            results.push_back(res);
            persist_result(res, bp);
            llama_batch_free(temp_batch_tgt);
            continue;
        }

        // Draft model prefill: inp[1..n-1] in chunks of n_batch_dft
        cb_data.data.clear();
        bool dft_prefill_ok = true;
        const int n_eagle_tokens = n_input - 1;
        for (int chunk_start = 0; chunk_start < n_eagle_tokens; chunk_start += n_batch_dft) {
            const int chunk_size = std::min(n_batch_dft, n_eagle_tokens - chunk_start);
            if (llama_decode_eagle(ctx_dft,
                    llama_batch_get_one(inp.data() + 1 + chunk_start, chunk_size),
                    sliced_data.data() + (size_t)chunk_start * hidden_dim) != 0) {
                fprintf(stderr, "  SKIP: draft model prefill failed at token %d/%d\n", chunk_start, n_eagle_tokens);
                dft_prefill_ok = false;
                break;
            }
        }

        if (!dft_prefill_ok) {
            common_sampler_free(smpl);
            res.error_message = "draft prefill failed";
            results.push_back(res);
            persist_result(res, bp);
            llama_batch_free(temp_batch_tgt);
            continue;
        }

        const auto t_enc_end = ggml_time_us();
        llama_batch_free(temp_batch_tgt);

        // ------ Decode state init ------
        int n_predict = 0;
        int n_drafted = 0;
        int n_accept  = 0;
        int n_past_tgt = n_input;
        int n_past_dft = n_input - 1;
        bool has_eos = false;
        bool prompt_failed = false;

        std::vector<seq_draft> drafts(n_seq_dft);
        std::vector<int> accepted_prefix_lengths;
        std::vector<int> step_output_lengths;
        std::vector<int> decoding_latencies;
        std::vector<int> verification_latencies;
        std::vector<float> T_d;
        int decode_step_index = 0;

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
                        for (int s = 0; s < n_seq_dft; ++s) {
                            if (!drafts[s].active) continue;
                            if (i_dft < (int)drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                                s_keep = s; accept = true;
                            } else {
                                drafts[s].active = false;
                            }
                        }
                    }

                    append_hidden_state_slice(temp2, backup_data, hidden_dim, drafts[s_keep].i_batch_tgt[i_dft]);
                    recompute.push_back(token_id);

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
            const int accepted_prefix_len = i_dft;
            const int step_output_len = (int) recompute.size();
            accepted_prefix_lengths.push_back(accepted_prefix_len);
            step_output_lengths.push_back(step_output_len);

            if (collect_vocab_stats) {
                record_verified_step_stats(analysis, bp.category, recompute, accepted_prefix_len);
            }
            if (shortlist_cfg.enabled) {
                record_coverage_step(analysis, bp.category, recompute, accepted_prefix_len, shortlist_cfg);
            }
            if (step_trace_ofs.is_open()) {
                const std::vector<bool> shortlist_hits = compute_shortlist_hit_flags(recompute, shortlist_cfg, bp.category);
                step_trace_ofs
                    << "{\"sample_index\":" << res.sample_index
                    << ",\"question_id\":" << bp.question_id
                    << ",\"category\":\"" << json_escape(normalize_category_key(bp.category)) << "\""
                    << ",\"decode_step_index\":" << decode_step_index
                    << ",\"accepted_prefix_len\":" << accepted_prefix_len
                    << ",\"step_output_len\":" << step_output_len
                    << ",\"verified_tokens\":";
                write_json_token_array(step_trace_ofs, recompute);
                step_trace_ofs << ",\"accepted_flags\":[";
                for (int idx = 0; idx < step_output_len; ++idx) {
                    if (idx > 0) {
                        step_trace_ofs << ",";
                    }
                    step_trace_ofs << (idx < accepted_prefix_len ? "true" : "false");
                }
                step_trace_ofs << "]";
                step_trace_ofs << ",\"bonus_index\":" << step_output_len
                               << ",\"bonus_token_id\":" << (step_output_len > 0 ? recompute.back() : LLAMA_TOKEN_NULL)
                               << ",\"shortlist_hit_flags\":";
                write_json_bool_array(step_trace_ofs, shortlist_hits);
                step_trace_ofs << ",\"eos\":" << (has_eos ? "true" : "false") << "}\n";
            }
            ++decode_step_index;

            for (int i = 0; i < n_seq_dft; i++)
                for (int j = 0; j < n_depth; j++)
                    scores[i][j] = 0.0f;

            backup_data = temp2;
            std::vector<float> temp3 = take_last_hidden_states(backup_data, hidden_dim, 1);
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

                if (prompt_failed) {
                    break;
                }

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
                    std::vector<float> temp4 = take_first_hidden_states(backup_data, hidden_dim, i_dft);
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

            if (prompt_failed || (params.n_predict >= 0 && n_predict > params.n_predict) || has_eos || dft_exhausted) {
                break;
            }

            const int n_ctx_dft = (int) llama_n_ctx(ctx_dft);
            if (n_past_dft + n_draft_max + n_depth >= n_ctx_dft - 2) {
                fprintf(stderr, "  SKIP: draft model context nearly full (n_past_dft=%d, n_ctx=%d)\n",
                        n_past_dft, n_ctx_dft);
                res.error_message = "draft context nearly full";
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

                    if (has_vocab_trim) {
                        for (size_t ci = 0; ci < cur_p->size; ++ci) {
                            const int idx = cur_p->data[ci].id;
                            if (idx >= 0 && idx < (int) dft_vocab_map.size()) {
                                cur_p->data[ci].id = dft_vocab_map[idx];
                            }
                        }
                    }

                    std::vector<int> sa(1, s);
                    append_hidden_state_slice(temp, cb_data.data, hidden_dim, drafts[s].i_batch_dft);

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
                            append_hidden_state_slice(temp, cb_data.data, hidden_dim, drafts[s].i_batch_dft);
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
                        const int proposed_position = (int) drafts[ss].tokens.size();
                        token_stats.draft_freq[id]++;
                        if (collect_vocab_stats) {
                            record_proposed_token_stats(analysis, bp.category, id, proposed_position);
                        }
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

        if (prompt_failed) {
            printf("\n");
            fprintf(stderr, "  --- output end ---\n");
            results.push_back(res);
            persist_result(res, bp);
            common_sampler_free(smpl);
            for (int s = 0; s < n_seq_dft; ++s) {
                if (drafts[s].smpl) { common_sampler_free(drafts[s].smpl); drafts[s].smpl = nullptr; }
            }
            continue;
        }

        // ---- Collect per-prompt stats ----
        res.n_predict = n_predict;
        res.n_drafted = n_drafted;
        res.n_accept  = n_accept;
        res.prefill_ms  = (t_enc_end - t_enc_start) / 1000.0;
        res.decode_ms   = (t_dec_end - t_dec_start) / 1000.0;
        res.prefill_tps = n_input / (res.prefill_ms / 1000.0);
        res.decode_tps  = n_predict > 0 ? n_predict / (res.decode_ms / 1000.0) : 0;
        res.decode_lat  = n_predict > 0 ? res.decode_ms / n_predict : 0;
        res.draft_len = !decoding_latencies.empty() ? (double) n_drafted / decoding_latencies.size() : 0;
        res.accept_ratio = n_drafted > 0 ? 100.0 * n_accept / n_drafted : 0;

        int n_steps = (int)decoding_latencies.size();
        res.avg_accepted_prefix_len = n_steps > 0 ? std::accumulate(accepted_prefix_lengths.begin(), accepted_prefix_lengths.end(), 0.0) / n_steps : 0;
        res.avg_step_output_len = n_steps > 0 ? std::accumulate(step_output_lengths.begin(), step_output_lengths.end(), 0.0) / n_steps : 0;
        res.avg_accept_len = res.avg_step_output_len;
        res.avg_draft_lat = !decoding_latencies.empty() ? std::accumulate(decoding_latencies.begin(), decoding_latencies.end(), 0.0) / decoding_latencies.size() : 0;
        res.avg_verify_lat = !verification_latencies.empty() ? std::accumulate(verification_latencies.begin(), verification_latencies.end(), 0.0) / verification_latencies.size() : 0;
        res.avg_td = !T_d.empty() ? std::accumulate(T_d.begin(), T_d.end(), 0.0) / T_d.size() : 0;
        res.success = true;

        printf("\n");
        fprintf(stderr, "  --- output end ---\n");
        fprintf(stderr, "  -> %d tokens | %.2f t/s | accept_len=%.2f | accept_ratio=%.1f%%\n",
                n_predict, res.decode_tps, res.avg_accept_len, res.accept_ratio);

        results.push_back(res);
        persist_result(res, bp);

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
    fprintf(stderr, "            %s Results (%zu prompts)\n", SPEC_BENCH_BACKEND_LABEL, results.size());
    fprintf(stderr, "============================================================\n");

    // Per-category aggregation
    std::map<std::string, std::vector<const bench_result *>> by_category;
    for (const auto & r : results) {
        by_category[r.category].push_back(&r);
    }

    std::vector<bench_summary> summaries;
    summaries.reserve(by_category.size() + 1);

    for (const auto & [cat, grp] : by_category) {
        const auto summary = summarize_results(cat, grp);
        print_summary_row(stderr, summary);
        summaries.push_back(summary);
    }

    fprintf(stderr, "------------------------------------------------------------\n");
    std::vector<const bench_result *> all_results;
    all_results.reserve(results.size());
    for (const auto & r : results) {
        all_results.push_back(&r);
    }
    const auto overall_summary = summarize_results("OVERALL", all_results);
    print_summary_row(stderr, overall_summary);
    summaries.push_back(overall_summary);

    int n_skipped = 0;
    for (const auto & r : results) { if (!r.success) n_skipped++; }
    if (n_skipped > 0) {
        fprintf(stderr, "  Skipped: %d prompts\n", n_skipped);
    }
    fprintf(stderr, "============================================================\n");

    // Write CSV (metrics only)
    {
        std::string csv_path = (std::filesystem::path(results_dir) / "results.csv").string();
        std::ofstream csv(csv_path);
        if (csv.is_open()) {
            csv << "sample_index,question_id,category,status,error_message,n_input,n_predict,n_drafted,n_accept,"
                   "prefill_ms,prefill_tps,decode_ms,decode_tps,decode_lat_ms,draft_len,accept_len,"
                   "accepted_prefix_len,step_output_len,accept_ratio,"
                   "avg_draft_ms,avg_verify_ms,avg_td_ms\n";
            for (const auto & r : results) {
                csv << r.sample_index << ","
                    << r.question_id << ","
                    << "\"" << csv_escape(r.category) << "\","
                    << "\"" << (r.success ? "success" : "skipped") << "\","
                    << "\"" << csv_escape(r.error_message) << "\","
                    << r.n_input << ","
                    << r.n_predict << ","
                    << r.n_drafted << ","
                    << r.n_accept << ","
                    << r.prefill_ms << ","
                    << r.prefill_tps << ","
                    << r.decode_ms << ","
                    << r.decode_tps << ","
                    << r.decode_lat << ","
                    << r.draft_len << ","
                    << r.avg_accept_len << ","
                    << r.avg_accepted_prefix_len << ","
                    << r.avg_step_output_len << ","
                    << r.accept_ratio << ","
                    << r.avg_draft_lat << ","
                    << r.avg_verify_lat << ","
                    << r.avg_td << "\n";
            }
            fprintf(stderr, "\nMetrics saved to: %s\n", csv_path.c_str());
        }
    }

    write_summary_csv(results_dir, summaries);

    // Write JSONL with outputs
    {
        std::string jsonl_path = (std::filesystem::path(results_dir) / "outputs.jsonl").string();
        std::ofstream ofs(jsonl_path);
        if (ofs.is_open()) {
            for (const auto & r : results) {
                ofs << "{\"sample_index\":" << r.sample_index
                    << ",\"question_id\":" << r.question_id
                    << ",\"category\":\"" << json_escape(r.category) << "\""
                    << ",\"status\":\"" << (r.success ? "success" : "skipped") << "\""
                    << ",\"decode_tps\":" << r.decode_tps
                    << ",\"draft_len\":" << r.draft_len
                    << ",\"accept_len\":" << r.avg_accept_len
                    << ",\"accepted_prefix_len\":" << r.avg_accepted_prefix_len
                    << ",\"step_output_len\":" << r.avg_step_output_len
                    << ",\"accept_ratio\":" << r.accept_ratio
                    << ",\"output\":\"" << json_escape(r.output_text) << "\""
                    << ",\"error_message\":\"" << json_escape(r.error_message) << "\"}\n";
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

        std::string freq_path = (std::filesystem::path(results_dir) / "token_freq.csv").string();
        std::ofstream freq_csv(freq_path);
        if (freq_csv.is_open()) {
            freq_csv << "token_id,token_text,draft_count,accepted_count,rejected_count,bonus_count,"
                     << "verified_total,accepted_total,bonus_total,proposed_total,accept_rate\n";

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
                         << lookup_total_count(analysis.overall.verified_total, r.id) << ","
                         << lookup_total_count(analysis.overall.accepted_total, r.id) << ","
                         << lookup_total_count(analysis.overall.bonus_total, r.id) << ","
                         << lookup_total_count(analysis.overall.proposed_total, r.id) << ","
                         << accept_rate << "\n";
            }
            fprintf(stderr, "\nToken frequency stats saved to: %s\n", freq_path.c_str());
        }
    }

    if (collect_vocab_stats) {
        write_accept_hist_csv(results_dir, analysis);
        write_token_pos_stats_csv(results_dir, ctx_tgt, analysis);
    }
    write_shortlist_coverage_csv(results_dir, analysis, shortlist_cfg);

    // ====================================================================
    // Cleanup
    // ====================================================================
    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);
    llama_backend_free();

    return 0;
}
