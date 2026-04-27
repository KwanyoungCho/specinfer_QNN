//EAGLE-2 QNN 구현 코드
//Target model runs on QNN NPU, Draft model runs on CPU/GPU via llama.cpp
//-ym-

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif
#include "gguf.h"
#include "llm_decode_runner.h"
#include "QNN/io_alloc.h"
#include "QNN/qnn_loader.h"
#include "QNN/qnn_qnnjson.h"
#include "QNN/qnn_tensor_util.h"
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
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

std::vector<size_t> TopK(const std::vector<float>& data, size_t k);

namespace {

constexpr int kExpectedHiddenDim = 4096;
constexpr int kDefaultSelectorTopK = 64;
constexpr size_t kSelectorTensorAlignment = 64;
constexpr int kDebugPrintTopN = 8;
constexpr int kOpenCLGatherMinRows = 512;

static int opencl_gather_padded_rows(int logical_rows) {
    return ((std::max(logical_rows, kOpenCLGatherMinRows) + 7) / 8) * 8;
}

static std::vector<int> parse_positive_int_list(const std::string & text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        const int value = std::stoi(item);
        if (value > 0) {
            values.push_back(value);
        }
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

static std::string join_int_list(const std::vector<int> & values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    return oss.str();
}

static std::vector<int> default_runtime_buckets(int max_rows) {
    static const int kDefaults[] = {
        512, 1024, 2048, 4096, 8192, 12288, 16384,
        24576, 32768, 49152, 65536, 98304, 128256,
    };

    std::vector<int> buckets;
    for (int value : kDefaults) {
        if (value <= max_rows) {
            buckets.push_back(opencl_gather_padded_rows(value));
        }
    }
    if (buckets.empty() || buckets.back() < max_rows) {
        buckets.push_back(opencl_gather_padded_rows(max_rows));
    }
    std::sort(buckets.begin(), buckets.end());
    buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
    return buckets;
}

static bool reduced_lm_head_backend_is_opencl(ggml_backend_t backend) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_is_opencl(backend);
#else
    (void) backend;
    return false;
#endif
}

static bool reduced_lm_head_opencl_supports_gather_rows_q4_0(
        ggml_backend_t backend,
        const ggml_tensor * src,
        int32_t dst_rows) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_supports_gather_rows_q4_0(backend, src, dst_rows);
#else
    (void) backend;
    (void) src;
    (void) dst_rows;
    return false;
#endif
}

static bool reduced_lm_head_opencl_gather_rows_q4_0(
        ggml_backend_t backend,
        const ggml_tensor * src,
        const int32_t * row_indices,
        int32_t n_rows,
        ggml_tensor * dst) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_gather_rows_q4_0(backend, src, row_indices, n_rows, dst);
#else
    (void) backend;
    (void) src;
    (void) row_indices;
    (void) n_rows;
    (void) dst;
    return false;
#endif
}

static bool reduced_lm_head_opencl_supports_top_k_f32(ggml_backend_t backend) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_supports_top_k_f32(backend);
#else
    (void) backend;
    return false;
#endif
}

static bool reduced_lm_head_opencl_supports_softmax_threshold_f32(ggml_backend_t backend) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_supports_softmax_threshold_f32(backend);
#else
    (void) backend;
    return false;
#endif
}

static bool reduced_lm_head_opencl_top_k_f32(
        ggml_backend_t backend,
        const float * scores,
        int32_t n_scores,
        int32_t top_k,
        int32_t * out_indices) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_top_k_f32(backend, scores, n_scores, top_k, out_indices);
#else
    (void) backend;
    (void) scores;
    (void) n_scores;
    (void) top_k;
    (void) out_indices;
    return false;
#endif
}

static bool reduced_lm_head_opencl_softmax_threshold_f32_to_device(
        ggml_backend_t backend,
        const float * scores,
        int32_t n_scores,
        float threshold,
        int32_t max_count,
        void ** out_device_indices,
        int32_t * out_count,
        int32_t * out_total_above) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_softmax_threshold_f32_to_device(
            backend,
            scores,
            n_scores,
            threshold,
            max_count,
            out_device_indices,
            out_count,
            out_total_above);
#else
    (void) backend;
    (void) scores;
    (void) n_scores;
    (void) threshold;
    (void) max_count;
    (void) out_device_indices;
    (void) out_count;
    (void) out_total_above;
    return false;
#endif
}

static bool reduced_lm_head_opencl_top_k_f32_to_device(
        ggml_backend_t backend,
        const float * scores,
        int32_t n_scores,
        int32_t top_k,
        void ** out_device_indices) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_top_k_f32_to_device(backend, scores, n_scores, top_k, out_device_indices);
#else
    (void) backend;
    (void) scores;
    (void) n_scores;
    (void) top_k;
    (void) out_device_indices;
    return false;
#endif
}

static bool reduced_lm_head_opencl_device_i32_buffer_copy_to_host(
        ggml_backend_t backend,
        void * device_buffer,
        int32_t count,
        int32_t * out_values) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_device_i32_buffer_copy_to_host(backend, device_buffer, count, out_values);
#else
    (void) backend;
    (void) device_buffer;
    (void) count;
    (void) out_values;
    return false;
#endif
}

static bool reduced_lm_head_opencl_device_i32_buffer_fill(
        ggml_backend_t backend,
        void * device_buffer,
        int32_t offset,
        int32_t count,
        int32_t value) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_device_i32_buffer_fill(backend, device_buffer, offset, count, value);
#else
    (void) backend;
    (void) device_buffer;
    (void) offset;
    (void) count;
    (void) value;
    return false;
#endif
}

static bool reduced_lm_head_opencl_device_i32_buffer_from_host(
        ggml_backend_t backend,
        const int32_t * values,
        int32_t count,
        void ** out_device_buffer) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_device_i32_buffer_from_host(backend, values, count, out_device_buffer);
#else
    (void) backend;
    (void) values;
    (void) count;
    (void) out_device_buffer;
    return false;
#endif
}

static bool reduced_lm_head_opencl_device_i32_buffer_write_from_host(
        ggml_backend_t backend,
        void * device_buffer,
        const int32_t * values,
        int32_t count) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_device_i32_buffer_write_from_host(backend, device_buffer, values, count);
#else
    (void) backend;
    (void) device_buffer;
    (void) values;
    (void) count;
    return false;
#endif
}

static bool reduced_lm_head_opencl_device_i32_buffer_sort_asc_inplace(
        ggml_backend_t backend,
        void * device_buffer,
        int32_t count) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_device_i32_buffer_sort_asc_inplace(backend, device_buffer, count);
#else
    (void) backend;
    (void) device_buffer;
    (void) count;
    return false;
#endif
}

static void reduced_lm_head_opencl_device_i32_buffer_free(
        ggml_backend_t backend,
        void * device_buffer) {
#ifdef GGML_USE_OPENCL
    ggml_backend_opencl_device_i32_buffer_free(backend, device_buffer);
#else
    (void) backend;
    (void) device_buffer;
#endif
}

static bool reduced_lm_head_opencl_gather_rows_q4_0_device_i32(
        ggml_backend_t backend,
        const ggml_tensor * src,
        void * device_row_indices,
        int32_t n_rows,
        ggml_tensor * dst) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_gather_rows_q4_0_device_i32(backend, src, device_row_indices, n_rows, dst);
#else
    (void) backend;
    (void) src;
    (void) device_row_indices;
    (void) n_rows;
    (void) dst;
    return false;
#endif
}

static bool reduced_lm_head_opencl_gather_rows_q4_0_device_i32_padded(
        ggml_backend_t backend,
        const ggml_tensor * src,
        void * device_row_indices,
        int32_t selected_rows,
        int32_t n_rows,
        int32_t pad_row_index,
        ggml_tensor * dst) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_gather_rows_q4_0_device_i32_padded(
            backend,
            src,
            device_row_indices,
            selected_rows,
            n_rows,
            pad_row_index,
            dst);
#else
    (void) backend;
    (void) src;
    (void) device_row_indices;
    (void) selected_rows;
    (void) n_rows;
    (void) pad_row_index;
    (void) dst;
    return false;
#endif
}

static bool reduced_lm_head_opencl_supports_indexed_mul_mat_q4_0(
        ggml_backend_t backend,
        const ggml_tensor * src,
        int32_t n_rows,
        int32_t batch_size) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_supports_indexed_mul_mat_q4_0(backend, src, n_rows, batch_size);
#else
    (void) backend;
    (void) src;
    (void) n_rows;
    (void) batch_size;
    return false;
#endif
}

static bool reduced_lm_head_opencl_indexed_mul_mat_q4_0(
        ggml_backend_t backend,
        const ggml_tensor * src,
        void * device_row_indices,
        int32_t n_rows,
        const ggml_tensor * hidden,
        ggml_tensor * dst) {
#ifdef GGML_USE_OPENCL
    return ggml_backend_opencl_indexed_mul_mat_q4_0(backend, src, device_row_indices, n_rows, hidden, dst);
#else
    (void) backend;
    (void) src;
    (void) device_row_indices;
    (void) n_rows;
    (void) hidden;
    (void) dst;
    return false;
#endif
}

struct DynamicSelectorConfig {
    int top_k = kDefaultSelectorTopK;
    int debug_log_level = 0;
    bool selector_softmax_threshold_enabled = false;
    float selector_softmax_threshold = 0.0f;
    std::string reduced_lmhead_gguf;
    std::string selector_ctx_dir;
    std::string selector_json_path;
    std::string selector_bin_path;
    std::string selector_backend_so;
    std::string selector_system_so;
    std::string selector_hot_vocab_json;
    bool dump_selector_scores = false;
    bool dump_reduced_logits = false;
    bool use_reduced_lmhead = false;
    bool force_packed_mul_mat = false;
    bool runtime_bucket_enabled = false;
    std::vector<int> runtime_buckets;
    float runtime_bucket_shrink_ratio = 0.5f;
    int runtime_bucket_shrink_patience = 8;
    bool selector_launch_after_recompute = false;
    int projector_cache_limit = 1;
    bool opencl_padded_device_ids = false;
    bool opencl_indexed_lmhead = true;
    bool opencl_indexed_lmhead_in_graph = false;
    bool selector_force_cpu_softmax_threshold = false;
    bool selector_force_opencl_softmax_threshold = false;
};

struct RuntimeRowBucketState {
    int current_rows = 0;
    int shrink_streak = 0;
};

static int choose_runtime_bucket_rows(
        int logical_rows,
        int max_rows,
        const DynamicSelectorConfig & config,
        RuntimeRowBucketState * state) {
    if (!config.runtime_bucket_enabled || state == nullptr || logical_rows <= 0 || max_rows <= 0) {
        return 0;
    }

    std::vector<int> buckets = config.runtime_buckets.empty()
            ? default_runtime_buckets(max_rows)
            : config.runtime_buckets;
    for (int & value : buckets) {
        value = opencl_gather_padded_rows(std::min(value, max_rows));
    }
    std::sort(buckets.begin(), buckets.end());
    buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());

    int target = opencl_gather_padded_rows(std::min(logical_rows, max_rows));
    for (int bucket : buckets) {
        if (bucket >= logical_rows) {
            target = bucket;
            break;
        }
    }
    target = std::max(target, opencl_gather_padded_rows(logical_rows));

    if (state->current_rows <= 0) {
        state->current_rows = target;
        state->shrink_streak = 0;
        return state->current_rows;
    }

    if (logical_rows > state->current_rows) {
        state->current_rows = target;
        state->shrink_streak = 0;
        return state->current_rows;
    }

    if (target < state->current_rows) {
        const float shrink_ratio = std::max(0.05f, std::min(config.runtime_bucket_shrink_ratio, 0.95f));
        if ((float) logical_rows <= (float) state->current_rows * shrink_ratio) {
            ++state->shrink_streak;
        } else {
            state->shrink_streak = 0;
        }

        if (state->shrink_streak >= std::max(1, config.runtime_bucket_shrink_patience)) {
            state->current_rows = target;
            state->shrink_streak = 0;
        }
    } else {
        state->shrink_streak = 0;
    }

    return state->current_rows;
}

struct OpenclI32BufferHandle {
    ggml_backend_t backend = nullptr;
    void * device_buffer = nullptr;
    int32_t count = 0;
    int32_t capacity = 0;

    ~OpenclI32BufferHandle() {
        if (device_buffer != nullptr) {
            reduced_lm_head_opencl_device_i32_buffer_free(backend, device_buffer);
            device_buffer = nullptr;
        }
    }
};

struct SelectorResult {
    std::vector<llama_token> token_ids;
    std::vector<float> scores;
    std::vector<int32_t> output_indices;
    std::shared_ptr<OpenclI32BufferHandle> opencl_output_indices_device;
};

struct SelectorExecProfile {
    int64_t total_us = 0;
    int64_t init_us = 0;
    int64_t input_write_us = 0;
    int64_t graph_execute_us = 0;
    int64_t output_read_us = 0;
    int64_t topk_us = 0;
};

struct RoundSelectionProfile {
    int64_t launch_us = 0;
    int64_t worker_dequeue_us = 0;
    int64_t task_start_us = 0;
    int64_t task_end_us = 0;
    SelectorExecProfile selector;
    int64_t shortlist_filter_us = 0;
    int64_t projector_init_us = 0;
};

struct ReducedDraftSamplingProfile {
    int64_t logits_compute_us = 0;
    int64_t logits_fetch_us = 0;
    int64_t sampler_apply_us = 0;
};

struct ReducedLmHeadContext {
    const ggml_tensor * tensor = nullptr;
    ggml_backend_t backend = nullptr; // borrowed from the draft context scheduler
    int hidden_dim = 0;
    int vocab_out = 0;
    bool has_vocab_trim = false;
    std::string model_path;
    std::string tensor_name;
    std::unordered_map<llama_token, int32_t> token_to_output_idx;
    bool output_idx_matches_token_id = false;
};

class ReducedLmHeadProjector;

struct RoundSelection {
    std::vector<llama_token> token_ids;
    std::vector<float> selector_scores;
    std::vector<int32_t> output_indices;
    std::shared_ptr<OpenclI32BufferHandle> opencl_output_indices_device;
    std::shared_ptr<ReducedLmHeadProjector> projector;
    std::vector<uint8_t> runtime_output_weights;
    ggml_tensor * runtime_output_source_tensor = nullptr;
    int runtime_output_rows = 0;
    bool runtime_output_borrowed = false;
    int dropped_token_count = 0;
};

struct RoundSelectionTaskResult {
    RoundSelection selection;
    RoundSelectionProfile profile;
};

struct RoundSelectionWorkerState {
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;
    bool has_pending_job = false;
    bool has_ready_result = false;
    uint64_t pending_job_id = 0;
    uint64_t ready_job_id = 0;
    int64_t pending_launch_us = 0;
    std::vector<float> hidden_input;
    RoundSelectionTaskResult result;
    std::exception_ptr error;
};

struct RoundSelectionWorkerGuard {
    RoundSelectionWorkerState * state = nullptr;
    std::thread * worker = nullptr;

    ~RoundSelectionWorkerGuard() {
        if (state == nullptr || worker == nullptr || !worker->joinable()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->stop = true;
            state->has_pending_job = false;
        }
        state->cv.notify_all();
        worker->join();
    }
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct DebugTrimmedLmHeadReference {
    bool enabled = false;
    std::string model_path;
    std::unique_ptr<ggml_context, GgmlContextDeleter> meta_ctx;
    ReducedLmHeadContext lm_head_ctx;
    std::vector<llama_token> cached_token_ids;
    RoundSelection cached_round_selection;
    bool logged_weight_compare_for_cache = false;
};

class CandidateSelector {
public:
    virtual ~CandidateSelector() = default;
    virtual bool warmup(std::string * error = nullptr) {
        (void) error;
        return true;
    }
    virtual SelectorResult run(
            const float * hidden,
            int hidden_dim,
            int top_k,
            SelectorExecProfile * profile = nullptr,
            bool need_scores = false) = 0;
    virtual const char * name() const = 0;
};

static bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string parent_dir(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return path.substr(0, 1);
    }
    return path.substr(0, pos);
}

static std::string base_name(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

static bool file_exists_readable(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

static bool load_token_id_json_array(
        const std::string & path,
        std::vector<llama_token> & token_ids,
        std::string & error) {
    token_ids.clear();

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "failed to open token id JSON: " + path;
        return false;
    }

    char ch = 0;
    bool in_number = false;
    bool negative = false;
    int64_t value = 0;
    auto finish_number = [&]() -> bool {
        if (!in_number) {
            return true;
        }
        if (negative || value > std::numeric_limits<int32_t>::max()) {
            std::ostringstream oss;
            oss << "invalid token id " << (negative ? -value : value) << " in " << path;
            error = oss.str();
            return false;
        }
        token_ids.push_back(static_cast<llama_token>(value));
        in_number = false;
        negative = false;
        value = 0;
        return true;
    };

    while (input.get(ch)) {
        if (ch == '-' && !in_number) {
            in_number = true;
            negative = true;
            value = 0;
            continue;
        }
        if (ch >= '0' && ch <= '9') {
            if (!in_number) {
                in_number = true;
                negative = false;
                value = 0;
            }
            value = value * 10 + static_cast<int64_t>(ch - '0');
            if (value > std::numeric_limits<int32_t>::max()) {
                error = "token id exceeds int32 range in " + path;
                return false;
            }
            continue;
        }
        if (!finish_number()) {
            return false;
        }
    }
    if (!finish_number()) {
        return false;
    }

    std::sort(token_ids.begin(), token_ids.end());
    token_ids.erase(std::unique(token_ids.begin(), token_ids.end()), token_ids.end());
    if (token_ids.empty()) {
        error = "token id JSON did not contain any ids: " + path;
        return false;
    }

    return true;
}

static bool resolve_selector_artifact_paths(
        const DynamicSelectorConfig & config,
        std::string & json_path,
        std::string & bin_path) {
    json_path.clear();
    bin_path.clear();

    if (!config.selector_json_path.empty()) {
        json_path = config.selector_json_path;
        bin_path = config.selector_bin_path.empty()
                ? parent_dir(json_path) + "/forward_0.bin"
                : config.selector_bin_path;
        return true;
    }

    if (!config.selector_ctx_dir.empty()) {
        if (ends_with(config.selector_ctx_dir, ".json")) {
            json_path = config.selector_ctx_dir;
            bin_path = config.selector_bin_path.empty()
                    ? parent_dir(json_path) + "/forward_0.bin"
                    : config.selector_bin_path;
            return true;
        }

        json_path = config.selector_ctx_dir + "/forward_0_json.json";
        bin_path = config.selector_bin_path.empty()
                ? config.selector_ctx_dir + "/forward_0.bin"
                : config.selector_bin_path;
        return true;
    }

    return false;
}

static size_t product_dims(const std::vector<uint32_t> & dims) {
    if (dims.empty()) {
        return 0;
    }

    size_t elements = 1;
    for (uint32_t dim : dims) {
        elements *= dim;
    }
    return elements;
}

static bool qnn_tensor_is_float16(const llama_qnn::QnnJsonTensorDesc & desc) {
    return desc.data_type.find("FLOAT_16") != std::string::npos;
}

static bool qnn_tensor_is_float32(const llama_qnn::QnnJsonTensorDesc & desc) {
    return desc.data_type.find("FLOAT_32") != std::string::npos;
}

static bool qnn_tensor_is_int32(const llama_qnn::QnnJsonTensorDesc & desc) {
    return desc.data_type.find("INT_32") != std::string::npos;
}

static bool qnn_tensor_is_int64(const llama_qnn::QnnJsonTensorDesc & desc) {
    return desc.data_type.find("INT_64") != std::string::npos;
}

static int qnn_loader_log_level_from_debug_level(int debug_log_level) {
    // Keep QNN DSP/driver logs quiet unless the user explicitly asks for verbose debugging.
    return debug_log_level >= 2 ? 5 : 1;
}

static void print_token_scores(
        const char * tag,
        const llama_context * ctx,
        const std::vector<llama_token> & token_ids,
        const std::vector<float> & scores,
        int max_items = kDebugPrintTopN) {
    const int limit = std::min<int>(max_items, std::min(token_ids.size(), scores.size()));
    LOG_INF("[%s] top %d entries\n", tag, limit);
    for (int i = 0; i < limit; ++i) {
        LOG_INF("  %2d: token=%6d score=%12.6f text='%s'\n",
                i,
                token_ids[i],
                scores[i],
                common_token_to_piece(ctx, token_ids[i]).c_str());
    }
}

static void print_candidate_distribution(
        const char * tag,
        const llama_context * ctx,
        const std::vector<llama_token_data> & candidates,
        int max_items = kDebugPrintTopN) {
    const int limit = std::min<int>(max_items, candidates.size());
    LOG_INF("[%s] top %d reduced logits\n", tag, limit);
    for (int i = 0; i < limit; ++i) {
        LOG_INF("  %2d: token=%6d logit=%12.6f prob=%10.6f text='%s'\n",
                i,
                candidates[i].id,
                candidates[i].logit,
                candidates[i].p,
                common_token_to_piece(ctx, candidates[i].id).c_str());
    }
}

static std::vector<float> extract_selector_input_hidden(const std::vector<float> & backup_data, int hidden_dim) {
    if (hidden_dim <= 0) {
        return {};
    }
    if (backup_data.size() < static_cast<size_t>(hidden_dim)) {
        return {};
    }
    if (backup_data.size() % hidden_dim != 0) {
        return {};
    }

    return std::vector<float>(backup_data.end() - hidden_dim, backup_data.end());
}

static SelectorResult topk_from_full_scores(const std::vector<float> & full_scores, int top_k) {
    SelectorResult result;
    if (full_scores.empty() || top_k <= 0) {
        return result;
    }

    const auto topk_indices = TopK(full_scores, top_k);
    result.token_ids.reserve(topk_indices.size());
    result.scores.reserve(topk_indices.size());
    for (size_t idx : topk_indices) {
        result.token_ids.push_back(static_cast<llama_token>(idx));
        result.scores.push_back(full_scores[idx]);
    }

    return result;
}

static SelectorResult softmax_threshold_from_full_scores(
        const std::vector<float> & full_scores,
        float threshold,
        int max_count,
        bool need_token_ids = true,
        bool need_scores = true,
        bool * truncated = nullptr,
        int * n_above_threshold = nullptr) {
    SelectorResult result;
    if (truncated != nullptr) {
        *truncated = false;
    }
    if (n_above_threshold != nullptr) {
        *n_above_threshold = 0;
    }
    if (full_scores.empty() || max_count <= 0 || threshold <= 0.0f) {
        return result;
    }

    float max_score = -std::numeric_limits<float>::infinity();
    int32_t max_idx = -1;
    for (size_t i = 0; i < full_scores.size(); ++i) {
        const float score = full_scores[i];
        if (!std::isfinite(score)) {
            continue;
        }
        if (score > max_score) {
            max_score = score;
            max_idx = static_cast<int32_t>(i);
        }
    }
    if (max_idx < 0) {
        return result;
    }

    double sum_exp = 0.0;
    for (const float score : full_scores) {
        if (!std::isfinite(score)) {
            continue;
        }
        sum_exp += std::exp(static_cast<double>(score - max_score));
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        result.output_indices.push_back(max_idx);
        if (need_token_ids) {
            result.token_ids.push_back(static_cast<llama_token>(max_idx));
        }
        if (need_scores) {
            result.scores.push_back(max_score);
        }
        return result;
    }

    const size_t output_limit = std::min<size_t>(static_cast<size_t>(max_count), full_scores.size());
    result.output_indices.reserve(output_limit);
    if (need_token_ids) {
        result.token_ids.reserve(output_limit);
    }
    if (need_scores) {
        result.scores.reserve(output_limit);
    }

    const double cutoff = static_cast<double>(max_score) + std::log(static_cast<double>(threshold) * sum_exp);
    int selected_count = 0;
    for (size_t i = 0; i < full_scores.size(); ++i) {
        const float score = full_scores[i];
        if (!std::isfinite(score)) {
            continue;
        }

        if (static_cast<double>(score) < cutoff) {
            continue;
        }

        ++selected_count;
        if (result.output_indices.size() >= output_limit) {
            continue;
        }

        result.output_indices.push_back(static_cast<int32_t>(i));
        if (need_token_ids) {
            result.token_ids.push_back(static_cast<llama_token>(i));
        }
        if (need_scores) {
            result.scores.push_back(score);
        }
    }

    if (n_above_threshold != nullptr) {
        *n_above_threshold = selected_count;
    }
    if (truncated != nullptr) {
        *truncated = selected_count > static_cast<int>(output_limit);
    }

    if (result.output_indices.empty()) {
        result.output_indices.push_back(max_idx);
        if (need_token_ids) {
            result.token_ids.push_back(static_cast<llama_token>(max_idx));
        }
        if (need_scores) {
            result.scores.push_back(max_score);
        }
    }

    return result;
}

class QnnRandomSelector : public CandidateSelector {
public:
    QnnRandomSelector(
            std::string json_path,
            std::string bin_path,
            std::string backend_so,
            std::string system_so,
            int qnn_log_level,
            ggml_backend_t opencl_topk_backend = nullptr,
            bool enable_opencl_topk = true,
            bool sort_topk_ids_for_gather = false,
            bool prefer_cpu_softmax_threshold = false,
            bool softmax_threshold_enabled = false,
            float softmax_threshold = 0.0f)
        : json_path_(std::move(json_path)),
          bin_path_(std::move(bin_path)),
          backend_so_(std::move(backend_so)),
          system_so_(std::move(system_so)),
          qnn_log_level_(qnn_log_level),
          opencl_topk_backend_(opencl_topk_backend),
          use_opencl_topk_(enable_opencl_topk && reduced_lm_head_opencl_supports_top_k_f32(opencl_topk_backend)),
          use_opencl_softmax_threshold_(
                  softmax_threshold_enabled &&
                  enable_opencl_topk &&
                  !prefer_cpu_softmax_threshold &&
                  reduced_lm_head_opencl_supports_softmax_threshold_f32(opencl_topk_backend)),
          output_idx_matches_token_id_(enable_opencl_topk),
          sort_topk_ids_for_gather_(sort_topk_ids_for_gather),
          softmax_threshold_enabled_(softmax_threshold_enabled),
          softmax_threshold_(softmax_threshold) {
    }

    SelectorResult run(
            const float * hidden,
            int hidden_dim,
            int top_k,
            SelectorExecProfile * profile = nullptr,
            bool need_scores = false) override {
        const auto run_start = ggml_time_us();
        if (!initialized_) {
            const auto init_start = ggml_time_us();
            if (!initialize()) {
                fprintf(stderr, "[selector-qnn] initialization failed: %s\n", error_.c_str());
                return {};
            }
            const auto init_end = ggml_time_us();
            if (profile != nullptr) {
                profile->init_us += (init_end - init_start);
            }
        }
        const auto input_write_start = ggml_time_us();
        if (!write_input(hidden, hidden_dim)) {
            fprintf(stderr, "[selector-qnn] input write failed: %s\n", error_.c_str());
            return {};
        }
        const auto input_write_end = ggml_time_us();
        if (profile != nullptr) {
            profile->input_write_us += (input_write_end - input_write_start);
        }

        const auto graph_execute_start = ggml_time_us();
        if (!loader_.execute_graph(0, graph_.graph_name, input_tensors_, output_tensors_)) {
            error_ = "QNN graphExecute failed for selector graph";
            fprintf(stderr, "[selector-qnn] %s\n", error_.c_str());
            return {};
        }
        const auto graph_execute_end = ggml_time_us();
        if (profile != nullptr) {
            profile->graph_execute_us += (graph_execute_end - graph_execute_start);
        }

        const auto output_read_start = ggml_time_us();
        if (!read_output(scratch_full_scores_, scratch_output_token_ids_)) {
            fprintf(stderr, "[selector-qnn] output read failed: %s\n", error_.c_str());
            return {};
        }
        const auto output_read_end = ggml_time_us();
        if (profile != nullptr) {
            profile->output_read_us += (output_read_end - output_read_start);
        }
        const std::vector<float> & full_scores = scratch_full_scores_;
        const std::vector<llama_token> & output_token_ids = scratch_output_token_ids_;

        const auto topk_start = ggml_time_us();
        SelectorResult result;
        if (output_mode_ == OutputMode::FULL_SCORES) {
            if (softmax_threshold_enabled_) {
                if (use_opencl_softmax_threshold_ && top_k > 0 && !full_scores.empty()) {
                    void * device_buffer = nullptr;
                    int32_t output_count = 0;
                    int32_t selected_above_threshold = 0;
                    if (reduced_lm_head_opencl_softmax_threshold_f32_to_device(
                                opencl_topk_backend_,
                                full_scores.data(),
                                static_cast<int32_t>(full_scores.size()),
                                softmax_threshold_,
                                top_k,
                                &device_buffer,
                                &output_count,
                                &selected_above_threshold) &&
                        device_buffer != nullptr &&
                        output_count > 0) {
                        auto handle = std::make_shared<OpenclI32BufferHandle>();
                        handle->backend = opencl_topk_backend_;
                        handle->device_buffer = device_buffer;
                        handle->count = output_count;
                        handle->capacity = top_k;

                        result.output_indices.resize(output_count, -1);
                        if (reduced_lm_head_opencl_device_i32_buffer_copy_to_host(
                                    opencl_topk_backend_,
                                    device_buffer,
                                    output_count,
                                    result.output_indices.data())) {
                            const int32_t padded_count = opencl_gather_padded_rows(output_count);
                            if (padded_count > output_count &&
                                padded_count <= top_k &&
                                !result.output_indices.empty() &&
                                reduced_lm_head_opencl_device_i32_buffer_fill(
                                        opencl_topk_backend_,
                                        device_buffer,
                                        output_count,
                                        padded_count - output_count,
                                        result.output_indices.back())) {
                                handle->count = padded_count;
                            }
                            result.opencl_output_indices_device = std::move(handle);
                            if (need_scores) {
                                result.scores.reserve(result.output_indices.size());
                                for (const int32_t token_id : result.output_indices) {
                                    result.scores.push_back(
                                            token_id >= 0 && token_id < static_cast<int32_t>(full_scores.size())
                                                    ? full_scores[(size_t) token_id]
                                                    : 0.0f);
                                }
                            }
                            if (selected_above_threshold > output_count &&
                                !softmax_threshold_truncation_logged_ &&
                                qnn_log_level_ >= 1) {
                                LOG_INF("[selector-qnn] softmax threshold selected %d ids above p>=%.8g; truncating to selector_top_k=%d\n",
                                        selected_above_threshold,
                                        static_cast<double>(softmax_threshold_),
                                        top_k);
                                softmax_threshold_truncation_logged_ = true;
                            }
                        }
                    }

                    if (result.output_indices.empty()) {
                        if (!opencl_softmax_threshold_fallback_logged_ && qnn_log_level_ >= 1) {
                            LOG_INF("[selector-qnn] OpenCL softmax-threshold helper failed; falling back to CPU threshold for later rounds\n");
                            opencl_softmax_threshold_fallback_logged_ = true;
                        }
                        use_opencl_softmax_threshold_ = false;
                    }
                }

                if (result.output_indices.empty() && result.token_ids.empty()) {
                    bool truncated = false;
                    int selected_above_threshold = 0;
                    const bool need_token_ids = !output_idx_matches_token_id_;
                    result = softmax_threshold_from_full_scores(
                            full_scores,
                            softmax_threshold_,
                            top_k,
                            need_token_ids,
                            need_scores,
                            &truncated,
                            &selected_above_threshold);
                    if (truncated && !softmax_threshold_truncation_logged_ && qnn_log_level_ >= 1) {
                        LOG_INF("[selector-qnn] softmax threshold selected %d ids above p>=%.8g; truncating to selector_top_k=%d\n",
                                selected_above_threshold,
                                static_cast<double>(softmax_threshold_),
                                top_k);
                        softmax_threshold_truncation_logged_ = true;
                    }
                }
            } else if (use_opencl_topk_ && top_k > 0 && !full_scores.empty()) {
                const int32_t output_count = std::min<int32_t>(top_k, static_cast<int32_t>(full_scores.size()));
                void * device_buffer = nullptr;
                if (reduced_lm_head_opencl_top_k_f32_to_device(
                            opencl_topk_backend_,
                            full_scores.data(),
                            static_cast<int32_t>(full_scores.size()),
                            top_k,
                            &device_buffer) &&
                    device_buffer != nullptr) {
                    auto handle = std::make_shared<OpenclI32BufferHandle>();
                        handle->backend = opencl_topk_backend_;
                        handle->device_buffer = device_buffer;
                        handle->count = output_count;
                        handle->capacity = output_count;

                    result.output_indices.resize(output_count, -1);
                    if (!reduced_lm_head_opencl_device_i32_buffer_copy_to_host(
                                opencl_topk_backend_,
                                device_buffer,
                                output_count,
                                result.output_indices.data())) {
                        result.output_indices.clear();
                    }

                    if (result.output_indices.empty()) {
                        handle.reset();
                    } else {
                        const bool ids_already_sorted =
                                std::is_sorted(result.output_indices.begin(), result.output_indices.end());
                        if (sort_topk_ids_for_gather_ && output_count > 1 && !ids_already_sorted) {
                            std::sort(result.output_indices.begin(), result.output_indices.end());
                            handle.reset();
                        }
                        if (handle != nullptr) {
                            result.opencl_output_indices_device = std::move(handle);
                        }
                    }
                    if (need_scores) {
                        result.scores.reserve(result.output_indices.size());
                        for (const int32_t token_id : result.output_indices) {
                            result.scores.push_back(
                                    token_id >= 0 && token_id < static_cast<int32_t>(full_scores.size())
                                            ? full_scores[(size_t) token_id]
                                            : 0.0f);
                        }
                    }
                    if (result.output_indices.empty()) {
                        if (!opencl_topk_fallback_logged_ && qnn_log_level_ >= 1) {
                            LOG_INF("[selector-qnn] OpenCL top-k result copy failed; falling back to CPU TopK for later rounds\n");
                            opencl_topk_fallback_logged_ = true;
                        }
                        use_opencl_topk_ = false;
                        result = topk_from_full_scores(full_scores, top_k);
                    }
                } else {
                    if (!opencl_topk_fallback_logged_ && qnn_log_level_ >= 1) {
                        LOG_INF("[selector-qnn] OpenCL top-k helper failed; falling back to CPU TopK for later rounds\n");
                        opencl_topk_fallback_logged_ = true;
                    }
                    use_opencl_topk_ = false;
                    result = topk_from_full_scores(full_scores, top_k);
                }
            } else {
                result = topk_from_full_scores(full_scores, top_k);
            }
        } else {
            if (top_k > static_cast<int>(output_token_ids.size()) && qnn_log_level_ >= 2) {
                LOG_INF("[selector-qnn] artifact returned %zu token ids, fewer than requested top_k=%d; using all returned ids\n",
                        output_token_ids.size(),
                        top_k);
            }
            const size_t limit = top_k > 0
                    ? std::min<size_t>(static_cast<size_t>(top_k), output_token_ids.size())
                    : output_token_ids.size();
            result.token_ids.assign(output_token_ids.begin(), output_token_ids.begin() + limit);
            if (need_scores) {
                result.scores.assign(limit, 0.0f);
            }
        }
        const auto topk_end = ggml_time_us();
        if (profile != nullptr) {
            profile->topk_us += (topk_end - topk_start);
            profile->total_us += (topk_end - run_start);
        }
        return result;
    }

    bool warmup(std::string * error = nullptr) override {
        if (initialized_) {
            return true;
        }
        if (initialize()) {
            return true;
        }
        if (error != nullptr) {
            *error = error_;
        }
        return false;
    }

    const char * name() const override {
        return "qnn_random";
    }

private:
    bool initialize() {
        std::ifstream bin_file(bin_path_, std::ios::binary);
        if (!bin_file) {
            error_ = "failed to open selector QNN context binary: " + bin_path_;
            return false;
        }
        bin_file.seekg(0, std::ios::end);
        const auto bin_size = static_cast<size_t>(bin_file.tellg());
        bin_file.seekg(0, std::ios::beg);
        context_binary_.resize(bin_size);
        bin_file.read(reinterpret_cast<char *>(context_binary_.data()), context_binary_.size());
        if (!bin_file) {
            error_ = "failed to read selector QNN context binary";
            return false;
        }

        if (!loader_.load(backend_so_, system_so_)) {
            error_ = "failed to load QNN shared libraries";
            return false;
        }
        loader_.set_log_level(qnn_log_level_);

        std::map<std::string, llama_qnn::QnnJsonGraphDesc> graphs;
        const bool has_json_metadata = llama_qnn::parse_qnn_json(json_path_, graphs);
        if (!has_json_metadata) {
            if (!llama_qnn::parse_qnn_binary_info(
                        loader_.handles().system_so_handle,
                        context_binary_.data(),
                        context_binary_.size(),
                        graphs)) {
                error_ = "failed to parse selector metadata from JSON or binary: json=" + json_path_ + " bin=" + bin_path_;
                return false;
            }
            if (qnn_log_level_ >= 5) {
                fprintf(stderr, "[selector-qnn] metadata loaded from binary '%s' (json optional)\n", bin_path_.c_str());
            }
        }

        auto graph_it = graphs.find("forward");
        if (graph_it == graphs.end()) {
            if (graphs.empty()) {
                error_ = "selector QNN artifact contains no graphs";
                return false;
            }
            graph_it = graphs.begin();
        }
        graph_ = graph_it->second;

        if (graph_.inputs.size() != 1 || graph_.outputs.size() != 1) {
            error_ = "selector QNN graph must have exactly one input and one output";
            return false;
        }

        if (!loader_.get_interface_provider(nullptr)) {
            error_ = "failed to obtain QNN interface provider";
            return false;
        }
        if (!loader_.create_backend_and_device()) {
            error_ = "failed to create QNN backend/device";
            return false;
        }
        loader_.enable_htp_performance_mode();

        if (!loader_.create_context_from_binary(context_binary_.data(), context_binary_.size())) {
            error_ = "failed to restore selector QNN context";
            return false;
        }
        if (!loader_.retrieve_graph(0, graph_.graph_name)) {
            error_ = "failed to retrieve selector QNN graph";
            return false;
        }

        alloc_.build_from_qnnjson(graph_);
        alloc_.allocate(kSelectorTensorAlignment);

        const auto & bindings = alloc_.bindings();

        for (const auto & input_desc : graph_.inputs) {
            auto it = bindings.find(input_desc.name);
            if (it == bindings.end() || it->second == nullptr) {
                error_ = "missing selector input allocation for tensor: " + input_desc.name;
                return false;
            }

            auto holder = std::make_unique<llama_qnn::QnnTensorHolder>();
            if (!holder->init_from_json(input_desc, it->second, input_desc.nbytes, true)) {
                error_ = "failed to initialize selector input tensor holder";
                return false;
            }
            input_holders_.push_back(std::move(holder));
        }

        for (const auto & output_desc : graph_.outputs) {
            auto it = bindings.find(output_desc.name);
            if (it == bindings.end() || it->second == nullptr) {
                error_ = "missing selector output allocation for tensor: " + output_desc.name;
                return false;
            }

            auto holder = std::make_unique<llama_qnn::QnnTensorHolder>();
            if (!holder->init_from_json(output_desc, it->second, output_desc.nbytes, false)) {
                error_ = "failed to initialize selector output tensor holder";
                return false;
            }
            output_holders_.push_back(std::move(holder));
        }

        input_tensors_.clear();
        input_tensors_.reserve(input_holders_.size());
        for (const auto & holder : input_holders_) {
            input_tensors_.push_back(holder->tensor());
        }
        output_tensors_.clear();
        output_tensors_.reserve(output_holders_.size());
        for (const auto & holder : output_holders_) {
            output_tensors_.push_back(holder->tensor());
        }

        input_desc_ = &graph_.inputs.front();
        output_desc_ = &graph_.outputs.front();
        input_buffer_ = bindings.at(input_desc_->name);
        output_buffer_ = bindings.at(output_desc_->name);
        input_num_elements_ = product_dims(input_desc_->dims);
        output_num_elements_ = product_dims(output_desc_->dims);

        if (qnn_tensor_is_float16(*output_desc_) || qnn_tensor_is_float32(*output_desc_)) {
            output_mode_ = OutputMode::FULL_SCORES;
            if (use_opencl_topk_ && qnn_log_level_ >= 1) {
                LOG_INF("[selector-qnn] full-score artifact detected; OpenCL top-k helper is available for shortlist projection\n");
            }
        } else if (qnn_tensor_is_int32(*output_desc_) || qnn_tensor_is_int64(*output_desc_)) {
            output_mode_ = OutputMode::TOPK_TOKEN_IDS;
            if (qnn_log_level_ >= 2) {
                LOG_INF("[selector-qnn] artifact output '%s' is %s with %zu elements; using device-side top-k ids directly\n",
                        output_desc_->name.c_str(),
                        output_desc_->data_type.c_str(),
                        output_num_elements_);
            }
        } else {
            error_ = "unsupported selector output datatype: " + output_desc_->data_type;
            return false;
        }

        initialized_ = true;
        return true;
    }

    bool write_input(const float * hidden, int hidden_dim) {
        if (input_desc_ == nullptr || input_buffer_ == nullptr) {
            error_ = "selector input buffer is not initialized";
            return false;
        }
        if (input_num_elements_ != static_cast<size_t>(hidden_dim)) {
            std::ostringstream oss;
            oss << "selector input dim mismatch: artifact expects " << input_num_elements_
                << " values but hidden_dim=" << hidden_dim;
            error_ = oss.str();
            return false;
        }

        if (qnn_tensor_is_float16(*input_desc_)) {
            ggml_fp32_to_fp16_row(hidden, reinterpret_cast<ggml_fp16_t *>(input_buffer_), hidden_dim);
            return true;
        }
        if (qnn_tensor_is_float32(*input_desc_)) {
            std::memcpy(input_buffer_, hidden, sizeof(float) * hidden_dim);
            return true;
        }

        error_ = "unsupported selector input datatype: " + input_desc_->data_type;
        return false;
    }

    bool read_output(std::vector<float> & output_scores, std::vector<llama_token> & output_token_ids) {
        if (output_desc_ == nullptr || output_buffer_ == nullptr) {
            error_ = "selector output buffer is not initialized";
            return false;
        }

        output_scores.clear();
        output_token_ids.clear();
        if (output_mode_ == OutputMode::FULL_SCORES) {
            output_scores.resize(output_num_elements_);
            if (qnn_tensor_is_float16(*output_desc_)) {
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(output_buffer_), output_scores.data(), output_num_elements_);
                return true;
            }
            if (qnn_tensor_is_float32(*output_desc_)) {
                std::memcpy(output_scores.data(), output_buffer_, sizeof(float) * output_num_elements_);
                return true;
            }
            error_ = "selector full-score output mode does not match datatype: " + output_desc_->data_type;
            return false;
        }

        output_token_ids.reserve(output_num_elements_);
        if (qnn_tensor_is_int32(*output_desc_)) {
            const int32_t * ids = reinterpret_cast<const int32_t *>(output_buffer_);
            for (size_t i = 0; i < output_num_elements_; ++i) {
                if (ids[i] < 0) {
                    continue;
                }
                output_token_ids.push_back(static_cast<llama_token>(ids[i]));
            }
            return true;
        }
        if (qnn_tensor_is_int64(*output_desc_)) {
            const int64_t * ids = reinterpret_cast<const int64_t *>(output_buffer_);
            for (size_t i = 0; i < output_num_elements_; ++i) {
                if (ids[i] < 0 || ids[i] > std::numeric_limits<llama_token>::max()) {
                    continue;
                }
                output_token_ids.push_back(static_cast<llama_token>(ids[i]));
            }
            return true;
        }

        error_ = "selector token-id output mode does not match datatype: " + output_desc_->data_type;
        return false;
    }

private:
    enum class OutputMode {
        FULL_SCORES,
        TOPK_TOKEN_IDS,
    };

    std::string json_path_;
    std::string bin_path_;
    std::string backend_so_;
    std::string system_so_;
    std::string error_;
    int qnn_log_level_ = 1;
    ggml_backend_t opencl_topk_backend_ = nullptr;
    bool use_opencl_topk_ = false;
    bool use_opencl_softmax_threshold_ = false;
    bool output_idx_matches_token_id_ = false;
    bool opencl_topk_fallback_logged_ = false;
    bool opencl_softmax_threshold_fallback_logged_ = false;
    bool sort_topk_ids_for_gather_ = false;
    bool softmax_threshold_enabled_ = false;
    bool softmax_threshold_truncation_logged_ = false;
    float softmax_threshold_ = 0.0f;

    bool initialized_ = false;
    llama_qnn::QnnLoader loader_;
    llama_qnn::QnnJsonGraphDesc graph_;
    llama_qnn::QNNIOAllocator alloc_;
    std::vector<std::unique_ptr<llama_qnn::QnnTensorHolder>> input_holders_;
    std::vector<std::unique_ptr<llama_qnn::QnnTensorHolder>> output_holders_;
    std::vector<Qnn_Tensor_t> input_tensors_;
    std::vector<Qnn_Tensor_t> output_tensors_;
    std::vector<uint8_t> context_binary_;
    std::vector<float> scratch_full_scores_;
    std::vector<llama_token> scratch_output_token_ids_;
    const llama_qnn::QnnJsonTensorDesc * input_desc_ = nullptr;
    const llama_qnn::QnnJsonTensorDesc * output_desc_ = nullptr;
    void * input_buffer_ = nullptr;
    void * output_buffer_ = nullptr;
    size_t input_num_elements_ = 0;
    size_t output_num_elements_ = 0;
    OutputMode output_mode_ = OutputMode::FULL_SCORES;
};

static std::unique_ptr<CandidateSelector> build_candidate_selector(
        const DynamicSelectorConfig & config,
        ggml_backend_t opencl_topk_backend,
        bool enable_opencl_topk,
        bool sort_topk_ids_for_gather) {
    std::string json_path;
    std::string bin_path;
    if (!resolve_selector_artifact_paths(config, json_path, bin_path)) {
        return nullptr;
    }
    const bool prefer_cpu_softmax_threshold =
            config.selector_force_cpu_softmax_threshold ||
            (config.opencl_indexed_lmhead && !config.selector_force_opencl_softmax_threshold);
    return std::make_unique<QnnRandomSelector>(
            json_path,
            bin_path,
            config.selector_backend_so,
            config.selector_system_so,
            qnn_loader_log_level_from_debug_level(config.debug_log_level),
            opencl_topk_backend,
            enable_opencl_topk,
            sort_topk_ids_for_gather,
            prefer_cpu_softmax_threshold,
            config.selector_softmax_threshold_enabled,
            config.selector_softmax_threshold);
}

static std::unordered_map<llama_token, int32_t> build_output_token_index_map(const llama_model * model) {
    std::unordered_map<llama_token, int32_t> token_to_output_idx;
    const uint32_t vocab_out = llama_model_n_vocab_out(model);
    token_to_output_idx.reserve(vocab_out);
    for (uint32_t i = 0; i < vocab_out; ++i) {
        // Match the trimmed/full sampler behavior: keep the first occurrence so
        // padded duplicate rows do not override the real LM head row.
        token_to_output_idx.emplace(llama_model_output_token_id(model, i), static_cast<int32_t>(i));
    }
    return token_to_output_idx;
}

static bool build_output_token_index_map_from_gguf(
        const gguf_context * gguf_ctx,
        const std::string & arch_name,
        int vocab_out,
        std::unordered_map<llama_token, int32_t> & token_to_output_idx,
        std::string & error) {
    token_to_output_idx.clear();
    if (vocab_out <= 0) {
        error = "trimmed GGUF reports non-positive output vocab size";
        return false;
    }

    const std::string vocab_map_key = arch_name + ".vocab_map";
    const int64_t vocab_map_key_id = gguf_find_key(gguf_ctx, vocab_map_key.c_str());
    if (vocab_map_key_id < 0) {
        token_to_output_idx.reserve(static_cast<size_t>(vocab_out));
        for (int32_t i = 0; i < vocab_out; ++i) {
            token_to_output_idx[static_cast<llama_token>(i)] = i;
        }
        return true;
    }

    const gguf_type arr_type = gguf_get_arr_type(gguf_ctx, vocab_map_key_id);
    const size_t arr_size = gguf_get_arr_n(gguf_ctx, vocab_map_key_id);
    if (arr_size < static_cast<size_t>(vocab_out)) {
        std::ostringstream oss;
        oss << "trimmed GGUF vocab_map size " << arr_size << " is smaller than output vocab " << vocab_out;
        error = oss.str();
        return false;
    }

    token_to_output_idx.reserve(static_cast<size_t>(vocab_out));
    const void * arr_data = gguf_get_arr_data(gguf_ctx, vocab_map_key_id);
    if (arr_data == nullptr) {
        error = "trimmed GGUF vocab_map data is null";
        return false;
    }

    // Keep the first occurrence of each token id so trailing padded duplicates do not
    // override the real row index.
    switch (arr_type) {
        case GGUF_TYPE_INT32: {
            const auto * values = static_cast<const int32_t *>(arr_data);
            for (int32_t i = 0; i < vocab_out; ++i) {
                token_to_output_idx.emplace(static_cast<llama_token>(values[i]), i);
            }
            return true;
        }
        case GGUF_TYPE_UINT32: {
            const auto * values = static_cast<const uint32_t *>(arr_data);
            for (int32_t i = 0; i < vocab_out; ++i) {
                token_to_output_idx.emplace(static_cast<llama_token>(values[i]), i);
            }
            return true;
        }
        default: {
            std::ostringstream oss;
            oss << "unsupported trimmed GGUF vocab_map array type: " << gguf_type_name(arr_type);
            error = oss.str();
            return false;
        }
    }
}

static bool prepare_debug_trimmed_lm_head_reference(
        const std::string & trimmed_model_path,
        const std::string & arch_name,
        const ReducedLmHeadContext & reference_lm_head_ctx,
        DebugTrimmedLmHeadReference & debug_ref,
        std::string & error) {
    ggml_context * meta_ctx_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_raw,
    };
    gguf_context * gguf_ctx = gguf_init_from_file(trimmed_model_path.c_str(), params);
    if (gguf_ctx == nullptr || meta_ctx_raw == nullptr) {
        if (meta_ctx_raw != nullptr) {
            ggml_free(meta_ctx_raw);
        }
        error = "failed to open trimmed GGUF metadata: " + trimmed_model_path;
        return false;
    }

    std::unique_ptr<ggml_context, GgmlContextDeleter> meta_ctx(meta_ctx_raw);
    const ggml_tensor * output = ggml_get_tensor(meta_ctx.get(), reference_lm_head_ctx.tensor_name.c_str());
    if (output == nullptr) {
        gguf_free(gguf_ctx);
        error = "trimmed GGUF is missing output tensor '" + reference_lm_head_ctx.tensor_name + "'";
        return false;
    }
    if (output->ne[2] != 1 || output->ne[3] != 1) {
        gguf_free(gguf_ctx);
        std::ostringstream oss;
        oss << "unexpected trimmed GGUF output tensor shape: ne=["
            << output->ne[0] << "," << output->ne[1] << "," << output->ne[2] << "," << output->ne[3] << "]";
        error = oss.str();
        return false;
    }
    if (output->ne[0] != reference_lm_head_ctx.hidden_dim) {
        gguf_free(gguf_ctx);
        std::ostringstream oss;
        oss << "trimmed GGUF hidden dim mismatch: got " << output->ne[0]
            << " expected " << reference_lm_head_ctx.hidden_dim;
        error = oss.str();
        return false;
    }
    if (output->type != reference_lm_head_ctx.tensor->type) {
        gguf_free(gguf_ctx);
        std::ostringstream oss;
        oss << "trimmed GGUF output type mismatch: got " << ggml_type_name(output->type)
            << " expected " << ggml_type_name(reference_lm_head_ctx.tensor->type);
        error = oss.str();
        return false;
    }

    ReducedLmHeadContext trimmed_ctx;
    trimmed_ctx.tensor = output;
    trimmed_ctx.backend = reference_lm_head_ctx.backend;
    trimmed_ctx.hidden_dim = static_cast<int>(output->ne[0]);
    trimmed_ctx.vocab_out = static_cast<int>(output->ne[1]);
    trimmed_ctx.has_vocab_trim = true;
    trimmed_ctx.model_path = trimmed_model_path;
    trimmed_ctx.tensor_name = ggml_get_name(output);
    trimmed_ctx.output_idx_matches_token_id = false;

    if (!build_output_token_index_map_from_gguf(
                gguf_ctx,
                arch_name,
                trimmed_ctx.vocab_out,
                trimmed_ctx.token_to_output_idx,
                error)) {
        gguf_free(gguf_ctx);
        return false;
    }

    gguf_free(gguf_ctx);

    debug_ref.enabled = true;
    debug_ref.model_path = trimmed_model_path;
    debug_ref.meta_ctx = std::move(meta_ctx);
    debug_ref.lm_head_ctx = std::move(trimmed_ctx);
    debug_ref.cached_token_ids.clear();
    debug_ref.cached_round_selection = RoundSelection{};
    debug_ref.logged_weight_compare_for_cache = false;
    return true;
}

static std::shared_ptr<std::vector<uint8_t>> load_gguf_tensor_snapshot(
        const std::string & model_path,
        const std::string & tensor_name,
        enum ggml_type expected_type,
        size_t expected_nbytes,
        std::string & error) {
    if (model_path.empty()) {
        error = "draft model path is empty";
        return nullptr;
    }
    if (tensor_name.empty()) {
        error = "GGUF tensor name is empty";
        return nullptr;
    }

    // Caching a full LM-head tensor snapshot is useful for tiny trimmed vocab
    // models, but keeping the full draft-model output tensor resident can add
    // hundreds of MiB of extra memory pressure and noticeably slow QNN target
    // verification on-device. Only cache relatively small snapshots.
    static constexpr size_t kMaxCachedSnapshotBytes = 128ull * 1024ull * 1024ull;
    const bool allow_cache = expected_nbytes <= kMaxCachedSnapshotBytes;

    static std::mutex gguf_snapshot_cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<std::vector<uint8_t>>> gguf_snapshot_cache;

    const std::string cache_key = model_path + "::" + tensor_name;
    if (allow_cache) {
        std::lock_guard<std::mutex> lock(gguf_snapshot_cache_mutex);
        auto it = gguf_snapshot_cache.find(cache_key);
        if (it != gguf_snapshot_cache.end()) {
            return it->second;
        }
    }

    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * gguf_ctx = gguf_init_from_file(model_path.c_str(), params);
    if (gguf_ctx == nullptr) {
        error = "failed to open GGUF metadata from: " + model_path;
        return nullptr;
    }

    const int64_t tensor_idx = gguf_find_tensor(gguf_ctx, tensor_name.c_str());
    if (tensor_idx < 0) {
        gguf_free(gguf_ctx);
        error = "tensor '" + tensor_name + "' not found in GGUF: " + model_path;
        return nullptr;
    }

    const enum ggml_type tensor_type = gguf_get_tensor_type(gguf_ctx, tensor_idx);
    const size_t tensor_size = gguf_get_tensor_size(gguf_ctx, tensor_idx);
    const size_t tensor_offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
    gguf_free(gguf_ctx);

    if (tensor_type != expected_type) {
        std::ostringstream oss;
        oss << "GGUF tensor type mismatch for '" << tensor_name << "': file="
            << ggml_type_name(tensor_type) << " expected=" << ggml_type_name(expected_type);
        error = oss.str();
        return nullptr;
    }
    if (tensor_size != expected_nbytes) {
        std::ostringstream oss;
        oss << "GGUF tensor size mismatch for '" << tensor_name << "': file="
            << tensor_size << " expected=" << expected_nbytes;
        error = oss.str();
        return nullptr;
    }

    auto tensor_bytes = std::make_shared<std::vector<uint8_t>>(tensor_size);
    std::ifstream input(model_path, std::ios::binary);
    if (!input) {
        error = "failed to open GGUF file for tensor snapshot: " + model_path;
        return nullptr;
    }

    input.seekg(static_cast<std::streamoff>(tensor_offset), std::ios::beg);
    if (!input) {
        error = "failed to seek GGUF tensor offset for: " + tensor_name;
        return nullptr;
    }

    input.read(reinterpret_cast<char *>(tensor_bytes->data()), static_cast<std::streamsize>(tensor_bytes->size()));
    if (!input) {
        error = "failed to read GGUF tensor bytes for: " + tensor_name;
        return nullptr;
    }

    if (allow_cache) {
        std::lock_guard<std::mutex> lock(gguf_snapshot_cache_mutex);
        gguf_snapshot_cache[cache_key] = tensor_bytes;
    }

    return tensor_bytes;
}

static ggml_backend_t resolve_tensor_backend(const llama_context * ctx, const ggml_tensor * tensor) {
    if (ctx == nullptr || tensor == nullptr || tensor->buffer == nullptr) {
        return nullptr;
    }

    const ggml_backend_buffer_type_t tensor_buft = ggml_backend_buffer_get_type(tensor->buffer);
    if (tensor_buft == nullptr) {
        return nullptr;
    }

    const ggml_backend_dev_t tensor_device = ggml_backend_buft_get_device(tensor_buft);
    if (tensor_device == nullptr) {
        return nullptr;
    }

    const ggml_backend_sched_t sched = ctx->get_sched();
    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (backend != nullptr && ggml_backend_get_device(backend) == tensor_device) {
            return backend;
        }
    }

    return nullptr;
}

static bool prepare_lm_head_context(
        const llama_model * model,
        const llama_context * ctx,
        const std::string & model_path,
        ReducedLmHeadContext & lm_head_ctx) {
    const ggml_tensor * output = model->output;
    if (output == nullptr) {
        fprintf(stderr, "[reduced-lmhead] output tensor is null\n");
        return false;
    }
    if (output->ne[2] != 1 || output->ne[3] != 1) {
        fprintf(stderr, "[reduced-lmhead] unexpected output tensor shape: ne=[%ld,%ld,%ld,%ld]\n",
                output->ne[0], output->ne[1], output->ne[2], output->ne[3]);
        return false;
    }

    lm_head_ctx.tensor = output;
    lm_head_ctx.backend = resolve_tensor_backend(ctx, output);
    lm_head_ctx.hidden_dim = static_cast<int>(output->ne[0]);
    lm_head_ctx.vocab_out = static_cast<int>(output->ne[1]);
    lm_head_ctx.has_vocab_trim = !model->vocab_map.empty();
    lm_head_ctx.model_path = model_path;
    lm_head_ctx.tensor_name = ggml_get_name(output);
    lm_head_ctx.token_to_output_idx = build_output_token_index_map(model);
    lm_head_ctx.output_idx_matches_token_id = !lm_head_ctx.has_vocab_trim;

    if (lm_head_ctx.backend == nullptr) {
        fprintf(stderr, "[reduced-lmhead] failed to resolve backend for output tensor\n");
        return false;
    }

    return true;
}

class ReducedLmHeadProjector {
public:
    ReducedLmHeadProjector() = default;

    ~ReducedLmHeadProjector() {
        clear_variants();
        clear_shared_weights();
    }

    ReducedLmHeadProjector(const ReducedLmHeadProjector &) = delete;
    ReducedLmHeadProjector & operator=(const ReducedLmHeadProjector &) = delete;

    bool initialize(
            const ReducedLmHeadContext & lm_head_ctx,
            const std::vector<int32_t> & output_indices,
            std::shared_ptr<OpenclI32BufferHandle> opencl_output_indices_device,
            bool emit_debug_logs,
            std::string & error,
            bool force_packed = false,
            bool require_host_packed = false,
            int requested_storage_rows = 0,
            bool use_opencl_padded_device_ids = false,
            bool use_opencl_indexed_lmhead = true,
            std::shared_ptr<OpenclI32BufferHandle> * reusable_upload_indices_device = nullptr) {
        if (lm_head_ctx.tensor == nullptr || lm_head_ctx.backend == nullptr) {
            error = "LM head tensor/backend is not initialized";
            return false;
        }
        if (output_indices.empty()) {
            error = "output_indices is empty";
            return false;
        }

        const ggml_backend_t prev_backend = backend_;
        const ggml_type prev_weight_type = weight_type_;
        const int prev_hidden_dim = hidden_dim_;
        const int prev_source_vocab_out = source_vocab_out_;
        const int prev_storage_shortlist_size = storage_shortlist_size_;
        const Mode prev_mode = mode_;

        packed_weights_.clear();
        output_indices_.clear();
        storage_output_indices_.clear();
        storage_to_output_pos_.clear();
        storage_order_identity_ = true;
        storage_logits_scratch_.clear();
        opencl_output_indices_device_.reset();
        opencl_gather_indices_device_.reset();
        opencl_gather_indices_padded_ = false;
        opencl_gather_selected_count_ = 0;
        opencl_gather_pad_row_ = -1;
        shared_weights_enqueued_ = false;
        shared_weights_ready_ = false;

        hidden_dim_ = lm_head_ctx.hidden_dim;
        shortlist_size_ = static_cast<int>(output_indices.size());
        backend_ = lm_head_ctx.backend;
        weight_type_ = lm_head_ctx.tensor->type;
        source_weights_tensor_ = const_cast<ggml_tensor *>(lm_head_ctx.tensor);
        source_vocab_out_ = lm_head_ctx.vocab_out;
        if (reusable_upload_indices_device != nullptr &&
            *reusable_upload_indices_device != nullptr &&
            (*reusable_upload_indices_device)->backend != backend_) {
            reusable_upload_indices_device->reset();
        }
        if (reusable_upload_indices_device != nullptr &&
            *reusable_upload_indices_device != nullptr) {
            opencl_uploaded_indices_device_ = *reusable_upload_indices_device;
        } else if (opencl_uploaded_indices_device_ != nullptr &&
                   opencl_uploaded_indices_device_->backend != backend_) {
            opencl_uploaded_indices_device_.reset();
        }
        output_indices_ = output_indices;
        opencl_output_indices_device_ = std::move(opencl_output_indices_device);
        storage_shortlist_size_ = shortlist_size_;
        storage_order_identity_ = true;
        emit_debug_logs_ = emit_debug_logs;

        for (const int32_t output_idx : output_indices_) {
            if (output_idx < 0 || output_idx >= lm_head_ctx.vocab_out) {
                std::ostringstream oss;
                oss << "invalid output_idx " << output_idx << " for vocab_out=" << lm_head_ctx.vocab_out;
                error = oss.str();
                return false;
            }
        }

        if (!require_host_packed) {
            if (!force_packed &&
                use_opencl_indexed_lmhead &&
                prepare_opencl_indexed_mul_mat(
                        emit_debug_logs,
                        requested_storage_rows,
                        use_opencl_padded_device_ids,
                        reusable_upload_indices_device)) {
                mode_ = Mode::OPENCL_INDEXED_MUL_MAT;
                const bool can_reuse_opencl_indexed =
                        prev_mode == Mode::OPENCL_INDEXED_MUL_MAT &&
                        prev_backend == backend_ &&
                        prev_weight_type == weight_type_ &&
                        prev_hidden_dim == hidden_dim_ &&
                        prev_source_vocab_out == source_vocab_out_ &&
                        prev_storage_shortlist_size == storage_shortlist_size_;
                if (!can_reuse_opencl_indexed) {
                    clear_variants();
                }
                clear_shared_weights();
                return true;
            }
            if (prepare_opencl_gather_mul_mat(
                        emit_debug_logs,
                        requested_storage_rows,
                        use_opencl_padded_device_ids,
                        reusable_upload_indices_device)) {
                mode_ = Mode::OPENCL_GATHER_MUL_MAT;
                const bool can_reuse_opencl_shared =
                        prev_mode == Mode::OPENCL_GATHER_MUL_MAT &&
                        prev_backend == backend_ &&
                        prev_weight_type == weight_type_ &&
                        prev_hidden_dim == hidden_dim_ &&
                        prev_source_vocab_out == source_vocab_out_ &&
                        prev_storage_shortlist_size == storage_shortlist_size_;
                if (!can_reuse_opencl_shared) {
                    clear_variants();
                    clear_shared_weights();
                }
                return true;
            }
            if (!force_packed && can_use_direct_mul_mat_id(emit_debug_logs)) {
                mode_ = Mode::DIRECT_MUL_MAT_ID;
                clear_variants();
                clear_shared_weights();
                return true;
            }
            if (force_packed) {
                if (can_use_gather_mul_mat(emit_debug_logs)) {
                    mode_ = Mode::GATHER_MUL_MAT;
                    clear_variants();
                    clear_shared_weights();
                    if (emit_debug_logs) {
                        LOG_INF("[reduced-lmhead] --force-packed-mul-mat: using gather_mul_mat (GPU get_rows + mul_mat)\n");
                    }
                    return true;
                }
                if (emit_debug_logs) {
                    LOG_INF("[reduced-lmhead] --force-packed-mul-mat: gather_mul_mat not available, falling back to packed_mul_mat\n");
                }
            }
        } else if (emit_debug_logs) {
            LOG_INF("[reduced-lmhead] materializing host-packed shortlist rows for runtime EAGLE output\n");
        }

        mode_ = Mode::PACKED_MUL_MAT;
        storage_shortlist_size_ = shortlist_size_;
        storage_output_indices_ = output_indices_;
        storage_order_identity_ = true;
        clear_variants();
        clear_shared_weights();

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(8, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * validate_ctx = ggml_init(params);
        if (validate_ctx == nullptr) {
            error = "ggml_init failed for reduced LM head graph";
            return false;
        }
        ggml_tensor * validate_weights = ggml_new_tensor_2d(validate_ctx, weight_type_, hidden_dim_, storage_shortlist_size_);
        ggml_tensor * validate_hidden  = ggml_new_tensor_2d(validate_ctx, GGML_TYPE_F32, hidden_dim_, 1);
        ggml_tensor * validate_logits  = ggml_mul_mat(validate_ctx, validate_weights, validate_hidden);

        const ggml_backend_dev_t backend_device = ggml_backend_get_device(backend_);
        if (backend_device == nullptr || !ggml_backend_dev_supports_op(backend_device, validate_logits)) {
            std::ostringstream oss;
            oss << "backend " << ggml_backend_name(backend_) << " does not support reduced LM head op";
            ggml_free(validate_ctx);
            error = oss.str();
            return false;
        }

        const size_t src_row_bytes = lm_head_ctx.tensor->nb[1];
        const size_t dst_row_bytes = validate_weights->nb[1];
        if (src_row_bytes == 0 || dst_row_bytes == 0) {
            ggml_free(validate_ctx);
            error = "invalid LM head row stride";
            return false;
        }
        if (src_row_bytes != dst_row_bytes) {
            std::ostringstream oss;
            oss << "LM head row stride mismatch: src=" << src_row_bytes << " dst=" << dst_row_bytes;
            ggml_free(validate_ctx);
            error = oss.str();
            return false;
        }
        ggml_free(validate_ctx);

        packed_weights_.resize(dst_row_bytes * output_indices_.size());
        bool gathered = false;

        // Fast path 1: direct pointer access (CPU-hosted buffer only).
        if (!gathered &&
            source_weights_tensor_ != nullptr &&
            source_weights_tensor_->data != nullptr &&
            source_weights_tensor_->buffer != nullptr &&
            ggml_backend_buffer_is_host(source_weights_tensor_->buffer) &&
            ggml_is_contiguous(source_weights_tensor_)) {
            const auto t_begin = std::chrono::steady_clock::now();
            const uint8_t * src = static_cast<const uint8_t *>(source_weights_tensor_->data);
            for (size_t row = 0; row < output_indices_.size(); ++row) {
                std::memcpy(
                        packed_weights_.data() + row * dst_row_bytes,
                        src + static_cast<size_t>(output_indices_[row]) * src_row_bytes,
                        src_row_bytes);
            }
            gathered = true;
            const auto t_end = std::chrono::steady_clock::now();
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] gathered %zu rows via direct memcpy in %.2f ms (%.2f KiB)\n",
                        output_indices_.size(),
                        std::chrono::duration<double, std::milli>(t_end - t_begin).count(),
                        packed_weights_.size() / 1024.0);
            }
        }

        // Fast path 2: bulk read entire tensor from backend buffer, then extract rows.
        if (!gathered &&
            source_weights_tensor_ != nullptr &&
            source_weights_tensor_->buffer != nullptr &&
            ggml_is_contiguous(source_weights_tensor_)) {
            const size_t full_nbytes = ggml_nbytes(source_weights_tensor_);
            const auto t_begin = std::chrono::steady_clock::now();
            std::vector<uint8_t> full_host(full_nbytes);
            ggml_backend_tensor_get(source_weights_tensor_, full_host.data(), 0, full_nbytes);
            for (size_t row = 0; row < output_indices_.size(); ++row) {
                std::memcpy(
                        packed_weights_.data() + row * dst_row_bytes,
                        full_host.data() + static_cast<size_t>(output_indices_[row]) * src_row_bytes,
                        src_row_bytes);
            }
            gathered = true;
            const auto t_end = std::chrono::steady_clock::now();
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] gathered %zu rows via bulk backend read in %.2f ms (%.2f MiB read, %.2f KiB packed)\n",
                        output_indices_.size(),
                        std::chrono::duration<double, std::milli>(t_end - t_begin).count(),
                        full_nbytes / (1024.0 * 1024.0),
                        packed_weights_.size() / 1024.0);
            }
        }

        // Fallback: read the full tensor from the GGUF file on disk.
        if (!gathered) {
            const auto t_begin = std::chrono::steady_clock::now();
            std::string snapshot_error;
            auto full_weights_host = load_gguf_tensor_snapshot(
                    lm_head_ctx.model_path,
                    lm_head_ctx.tensor_name,
                    weight_type_,
                    ggml_nbytes(lm_head_ctx.tensor),
                    snapshot_error);
            const auto t_end = std::chrono::steady_clock::now();

            if (!full_weights_host) {
                error = "failed to load LM head tensor snapshot from GGUF: " + snapshot_error;
                return false;
            }

            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] fallback: loaded full tensor from GGUF in %.2f ms (%.2f MiB)\n",
                        std::chrono::duration<double, std::milli>(t_end - t_begin).count(),
                        full_weights_host->size() / (1024.0 * 1024.0));
            }

            for (size_t row = 0; row < output_indices_.size(); ++row) {
                std::memcpy(
                        packed_weights_.data() + row * dst_row_bytes,
                        full_weights_host->data() + static_cast<size_t>(output_indices_[row]) * src_row_bytes,
                        src_row_bytes);
            }
        }
        return true;
    }

    bool prepare_for_compute(std::string & error) {
        if (mode_ != Mode::OPENCL_GATHER_MUL_MAT) {
            return true;
        }
        return ensure_opencl_gather_weights_ready(error);
    }

    // Enqueue the OpenCL gather kernel without blocking on completion.
    // Lets the gather overlap with unrelated backend work (e.g. QNN draft forward).
    // The caller must eventually invoke prepare_for_compute (or compute_logits*) to
    // synchronize before the gathered rows are read by a graph.
    bool prewarm_async(std::string & error) {
        if (mode_ != Mode::OPENCL_GATHER_MUL_MAT) {
            return true;
        }
        return ensure_opencl_gather_weights_enqueued(error);
    }

    bool compute_logits(const float * hidden, std::vector<float> & logits_out, double * compute_ms, std::string & error) {
        return compute_logits_batch(hidden, 1, logits_out, compute_ms, error);
    }

    bool compute_logits_batch(
            const float * hidden_batch,
            int batch_size,
            std::vector<float> & logits_out,
            double * compute_ms,
            std::string & error) {
        if (hidden_batch == nullptr) {
            error = "hidden pointer is null";
            return false;
        }
        if (batch_size <= 0) {
            error = "batch_size must be positive";
            return false;
        }

        // gather_mul_mat: ggml_get_rows dequantizes to F32; a full shortlist (e.g. 32k × hidden) is ~512 MiB
        // per intermediate and often exceeds device buffer limits — run in chunks.
        if (mode_ == Mode::GATHER_MUL_MAT) {
            logits_out.assign(static_cast<size_t>(shortlist_size_) * batch_size, 0.0f);
            std::vector<float> partial;
            partial.resize(static_cast<size_t>(std::min(kGatherMulMatChunkLen, shortlist_size_)) * batch_size);
            const auto t_begin = std::chrono::steady_clock::now();
            for (int off = 0; off < shortlist_size_; off += kGatherMulMatChunkLen) {
                const int clen = std::min(kGatherMulMatChunkLen, shortlist_size_ - off);
                GraphVariant * variant = get_or_create_variant(batch_size, clen, error);
                if (variant == nullptr) {
                    return false;
                }
                if (variant->hidden_tensor == nullptr || variant->logits_tensor == nullptr || variant->graph == nullptr) {
                    error = "projector is not initialized";
                    return false;
                }
                ggml_backend_tensor_set(
                        variant->ids_tensor,
                        output_indices_.data() + off,
                        0,
                        sizeof(int32_t) * static_cast<size_t>(clen));
                ggml_backend_tensor_set(
                        variant->hidden_tensor,
                        hidden_batch,
                        0,
                        sizeof(float) * hidden_dim_ * batch_size);
                const ggml_status status = ggml_backend_graph_compute(backend_, variant->graph);
                if (status != GGML_STATUS_SUCCESS) {
                    std::ostringstream oss;
                    oss << "ggml_backend_graph_compute failed with status " << status;
                    error = oss.str();
                    return false;
                }
                ggml_backend_tensor_get(
                        variant->logits_tensor,
                        partial.data(),
                        0,
                        sizeof(float) * static_cast<size_t>(clen) * batch_size);
                for (int b = 0; b < batch_size; ++b) {
                    std::memcpy(
                            logits_out.data() + static_cast<size_t>(b) * shortlist_size_ + off,
                            partial.data() + static_cast<size_t>(b) * clen,
                            sizeof(float) * static_cast<size_t>(clen));
                }
            }
            const auto t_end = std::chrono::steady_clock::now();
            if (compute_ms != nullptr) {
                *compute_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
            }
            return true;
        }

        if (mode_ == Mode::OPENCL_GATHER_MUL_MAT) {
            if (!prepare_for_compute(error)) {
                return false;
            }
        }

        if (mode_ == Mode::OPENCL_INDEXED_MUL_MAT) {
            if (opencl_gather_indices_device_ == nullptr ||
                opencl_gather_indices_device_->device_buffer == nullptr ||
                opencl_gather_indices_device_->count < storage_shortlist_size_) {
                error = "OpenCL indexed reduced LM head is missing device row ids";
                return false;
            }

            GraphVariant * variant = get_or_create_variant(batch_size, storage_shortlist_size_, error);
            if (variant == nullptr) {
                return false;
            }
            if (variant->hidden_tensor == nullptr || variant->logits_tensor == nullptr) {
                error = "OpenCL indexed projector variant is not initialized";
                return false;
            }

            ggml_backend_tensor_set(
                    variant->hidden_tensor,
                    hidden_batch,
                    0,
                    sizeof(float) * hidden_dim_ * batch_size);

            const auto t_begin = std::chrono::steady_clock::now();
            const bool ok = reduced_lm_head_opencl_indexed_mul_mat_q4_0(
                    backend_,
                    source_weights_tensor_,
                    opencl_gather_indices_device_->device_buffer,
                    storage_shortlist_size_,
                    variant->hidden_tensor,
                    variant->logits_tensor);
            const auto t_end = std::chrono::steady_clock::now();
            if (compute_ms != nullptr) {
                *compute_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
            }
            if (!ok) {
                error = "OpenCL indexed reduced LM head matmul failed";
                return false;
            }

            logits_out.resize(static_cast<size_t>(shortlist_size_) * batch_size);
            storage_logits_scratch_.resize(static_cast<size_t>(storage_shortlist_size_) * batch_size);
            ggml_backend_tensor_get(
                    variant->logits_tensor,
                    storage_logits_scratch_.data(),
                    0,
                    storage_logits_scratch_.size() * sizeof(float));
            reorder_storage_logits_to_output_order(storage_logits_scratch_.data(), batch_size, logits_out);
            return true;
        }

        GraphVariant * variant = get_or_create_variant(batch_size, storage_shortlist_size_, error);
        if (variant == nullptr) {
            return false;
        }
        if (variant->hidden_tensor == nullptr || variant->logits_tensor == nullptr || variant->graph == nullptr) {
            error = "projector is not initialized";
            return false;
        }

        ggml_backend_tensor_set(
                variant->hidden_tensor,
                hidden_batch,
                0,
                sizeof(float) * hidden_dim_ * batch_size);

        const auto t_begin = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend_, variant->graph);
        const auto t_end = std::chrono::steady_clock::now();
        if (compute_ms != nullptr) {
            *compute_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
        }
        if (status != GGML_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "ggml_backend_graph_compute failed with status " << status;
            error = oss.str();
            return false;
        }

        logits_out.resize(static_cast<size_t>(shortlist_size_) * batch_size);
        if (mode_ == Mode::OPENCL_GATHER_MUL_MAT) {
            storage_logits_scratch_.resize(static_cast<size_t>(storage_shortlist_size_) * batch_size);
            ggml_backend_tensor_get(
                    variant->logits_tensor,
                    storage_logits_scratch_.data(),
                    0,
                    storage_logits_scratch_.size() * sizeof(float));
            reorder_storage_logits_to_output_order(storage_logits_scratch_.data(), batch_size, logits_out);
        } else if (storage_shortlist_size_ == shortlist_size_) {
            ggml_backend_tensor_get(
                    variant->logits_tensor,
                    logits_out.data(),
                    0,
                    logits_out.size() * sizeof(float));
        } else {
            std::vector<float> padded_logits(static_cast<size_t>(storage_shortlist_size_) * batch_size);
            ggml_backend_tensor_get(
                    variant->logits_tensor,
                    padded_logits.data(),
                    0,
                    padded_logits.size() * sizeof(float));
            for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                std::memcpy(
                        logits_out.data() + static_cast<size_t>(batch_idx) * shortlist_size_,
                        padded_logits.data() + static_cast<size_t>(batch_idx) * storage_shortlist_size_,
                        sizeof(float) * static_cast<size_t>(shortlist_size_));
            }
        }
        return true;
    }

    const std::vector<uint8_t> & packed_weights() const {
        return packed_weights_;
    }

    bool runtime_output_copy_source_tensor(
            int runtime_output_rows,
            ggml_tensor ** out_tensor,
            std::string & error) {
        if (out_tensor == nullptr) {
            error = "runtime output tensor destination is null";
            return false;
        }
        *out_tensor = nullptr;

        if (mode_ != Mode::OPENCL_GATHER_MUL_MAT) {
            return true;
        }
        if (runtime_output_rows != storage_shortlist_size_) {
            return true;
        }
        if (shared_weights_tensor_ == nullptr) {
            if (!ensure_opencl_gather_weights_enqueued(error)) {
                return false;
            }
        } else if (!ensure_opencl_gather_weights_enqueued(error)) {
            return false;
        }
        if (!storage_to_output_pos_.empty()) {
            for (int storage_pos = 0; storage_pos < shortlist_size_; ++storage_pos) {
                if (storage_to_output_pos_[storage_pos] != storage_pos) {
                    return true;
                }
            }
        }

        *out_tensor = shared_weights_tensor_;
        return true;
    }

    bool export_packed_weights(std::vector<uint8_t> & packed_out, std::string & error) {
        if (!packed_weights_.empty()) {
            packed_out = packed_weights_;
            return true;
        }

        if (mode_ != Mode::OPENCL_GATHER_MUL_MAT) {
            error = std::string("packed weight export is unsupported for mode ") + mode_name();
            return false;
        }

        if (!prepare_for_compute(error)) {
            return false;
        }
        if (shared_weights_tensor_ == nullptr) {
            error = "shared OpenCL gathered weights tensor is not initialized";
            return false;
        }
        if (source_weights_tensor_ == nullptr || source_weights_tensor_->nb[1] == 0) {
            error = "source LM head tensor row stride is unavailable";
            return false;
        }

        const size_t row_bytes = source_weights_tensor_->nb[1];
        std::vector<uint8_t> storage_weights(row_bytes * static_cast<size_t>(storage_shortlist_size_));
        ggml_backend_tensor_get(
                shared_weights_tensor_,
                storage_weights.data(),
                0,
                storage_weights.size());

        packed_out.resize(row_bytes * static_cast<size_t>(shortlist_size_));
        for (int storage_pos = 0; storage_pos < shortlist_size_; ++storage_pos) {
            const int output_pos = storage_to_output_pos_.empty()
                    ? storage_pos
                    : storage_to_output_pos_[storage_pos];
            std::memcpy(
                    packed_out.data() + row_bytes * static_cast<size_t>(output_pos),
                    storage_weights.data() + row_bytes * static_cast<size_t>(storage_pos),
                    row_bytes);
        }

        return true;
    }

    ggml_type weight_type() const {
        return weight_type_;
    }

    const char * mode_name() const {
        switch (mode_) {
            case Mode::DIRECT_MUL_MAT_ID:
                return "mul_mat_id_direct";
            case Mode::PACKED_MUL_MAT:
                return "packed_mul_mat";
            case Mode::OPENCL_GATHER_MUL_MAT:
                return "opencl_gather_mul_mat";
            case Mode::OPENCL_INDEXED_MUL_MAT:
                return "opencl_indexed_mul_mat";
            case Mode::GATHER_MUL_MAT:
                return "gather_mul_mat";
        }
        return "unknown";
    }

    int runtime_output_row_capacity() const {
        return (mode_ == Mode::OPENCL_GATHER_MUL_MAT || mode_ == Mode::OPENCL_INDEXED_MUL_MAT)
                ? storage_shortlist_size_
                : shortlist_size_;
    }

    bool is_opencl_indexed_mode() const {
        return mode_ == Mode::OPENCL_INDEXED_MUL_MAT;
    }

    const std::vector<int32_t> & runtime_output_indices() const {
        return storage_output_indices_;
    }

private:
    void clear_variants() {
        variants_.clear();
    }

    void clear_shared_weights() {
        shared_weights_tensor_ = nullptr;
        shared_weights_ready_ = false;
        shared_weights_enqueued_ = false;
        if (shared_weights_buffer_ != nullptr) {
            ggml_backend_buffer_free(shared_weights_buffer_);
            shared_weights_buffer_ = nullptr;
        }
        if (shared_weights_ctx_ != nullptr) {
            ggml_free(shared_weights_ctx_);
            shared_weights_ctx_ = nullptr;
        }
    }

    enum class Mode {
        PACKED_MUL_MAT,
        DIRECT_MUL_MAT_ID,
        OPENCL_GATHER_MUL_MAT,
        OPENCL_INDEXED_MUL_MAT,
        GATHER_MUL_MAT,
    };

    struct GraphVariant {
        ggml_backend_buffer_t buffer = nullptr;
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * weights_tensor = nullptr;
        ggml_tensor * hidden_tensor = nullptr;
        ggml_tensor * ids_tensor = nullptr;
        ggml_tensor * gathered_tensor = nullptr;
        ggml_tensor * logits_tensor = nullptr;

        ~GraphVariant() {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
            }
            if (ctx != nullptr) {
                ggml_free(ctx);
            }
        }
    };

    bool ensure_opencl_gather_weights_enqueued(std::string & error) {
        if (mode_ != Mode::OPENCL_GATHER_MUL_MAT) {
            return true;
        }
        if (shared_weights_enqueued_) {
            return true;
        }

        if (shared_weights_tensor_ == nullptr) {
            ggml_init_params params = {
                /* .mem_size   = */ ggml_tensor_overhead() * 4,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };
            shared_weights_ctx_ = ggml_init(params);
            if (shared_weights_ctx_ == nullptr) {
                error = "ggml_init failed for shared reduced LM head weights";
                return false;
            }

            shared_weights_tensor_ = ggml_new_tensor_2d(
                    shared_weights_ctx_,
                    weight_type_,
                    hidden_dim_,
                    storage_shortlist_size_);
            if (shared_weights_tensor_ == nullptr) {
                error = "failed to create shared reduced LM head weights tensor";
                return false;
            }
            ggml_set_name(shared_weights_tensor_, "reduced_lm_head_opencl_gather");
        }

        if (shared_weights_buffer_ == nullptr) {
            shared_weights_buffer_ = ggml_backend_alloc_ctx_tensors(shared_weights_ctx_, backend_);
            if (shared_weights_buffer_ == nullptr) {
                error = "failed to allocate shared reduced LM head weights buffer";
                return false;
            }
            ggml_backend_buffer_set_usage(shared_weights_buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

            if (emit_debug_logs_) {
                LOG_INF("[reduced-lmhead] shared OpenCL gather weights buffer allocated rows=%d\n",
                        storage_shortlist_size_);
            }
        }

        if (emit_debug_logs_) {
            LOG_INF("[reduced-lmhead] OpenCL gather rows start logical=%d storage=%d\n",
                    shortlist_size_,
                    storage_shortlist_size_);
        }
        bool gather_ok = false;
        if (opencl_gather_indices_device_ != nullptr &&
            opencl_gather_indices_device_->device_buffer != nullptr) {
            if (opencl_gather_indices_padded_) {
                gather_ok = reduced_lm_head_opencl_gather_rows_q4_0_device_i32_padded(
                        backend_,
                        source_weights_tensor_,
                        opencl_gather_indices_device_->device_buffer,
                        opencl_gather_selected_count_,
                        storage_shortlist_size_,
                        opencl_gather_pad_row_,
                        shared_weights_tensor_);
            } else if (opencl_gather_indices_device_->count == storage_shortlist_size_) {
                gather_ok = reduced_lm_head_opencl_gather_rows_q4_0_device_i32(
                        backend_,
                        source_weights_tensor_,
                        opencl_gather_indices_device_->device_buffer,
                        storage_shortlist_size_,
                        shared_weights_tensor_);
            }
        }
        if (!gather_ok) {
            gather_ok = reduced_lm_head_opencl_gather_rows_q4_0(
                    backend_,
                    source_weights_tensor_,
                    storage_output_indices_.data(),
                    storage_shortlist_size_,
                    shared_weights_tensor_);
        }
        if (!gather_ok) {
            error = "failed to gather reduced LM head rows on OpenCL device";
            return false;
        }
        if (emit_debug_logs_) {
            LOG_INF("[reduced-lmhead] OpenCL gather rows enqueued (async)\n");
        }
        shared_weights_enqueued_ = true;
        return true;
    }

    bool ensure_opencl_gather_weights_ready(std::string & error) {
        if (mode_ != Mode::OPENCL_GATHER_MUL_MAT) {
            return true;
        }
        if (shared_weights_ready_) {
            return true;
        }
        if (!ensure_opencl_gather_weights_enqueued(error)) {
            return false;
        }
        if (emit_debug_logs_) {
            LOG_INF("[reduced-lmhead] OpenCL gather rows synchronizing\n");
        }
        ggml_backend_synchronize(backend_);
        shared_weights_ready_ = true;
        if (emit_debug_logs_) {
            LOG_INF("[reduced-lmhead] OpenCL gather rows completed\n");
        }
        return true;
    }

    bool can_use_direct_mul_mat_id(bool emit_debug_logs) const {
        // The direct mul_mat_id projector is promising for avoiding shortlist packing,
        // but on the current Adreno/OpenCL draft path it can produce logits that do not
        // match the reference LM head closely enough, which collapses acceptance.
        // Keep the safe packed-mul_mat fallback on OpenCL until the direct path is
        // validated end-to-end on this backend.
        const char * backend_name = backend_ != nullptr ? ggml_backend_name(backend_) : "unknown";
        if (backend_name != nullptr && std::string(backend_name).find("OpenCL") != std::string::npos) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] mul_mat_id direct disabled on backend %s due to correctness regression; using packed fallback\n",
                        backend_name);
            }
            return false;
        }

        if (source_weights_tensor_ == nullptr || source_weights_tensor_->buffer == nullptr) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] mul_mat_id direct disabled: LM head tensor has no live backend buffer\n");
            }
            return false;
        }
        if (!ggml_is_contiguous(source_weights_tensor_)) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] mul_mat_id direct disabled: LM head tensor is not contiguous\n");
            }
            return false;
        }

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(8, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * validate_ctx = ggml_init(params);
        if (validate_ctx == nullptr) {
            return false;
        }

        ggml_tensor * validate_weights_2d = ggml_new_tensor_2d(validate_ctx, weight_type_, hidden_dim_, source_vocab_out_);
        ggml_tensor * validate_weights_3d = ggml_reshape_3d(validate_ctx, validate_weights_2d, hidden_dim_, 1, source_vocab_out_);
        ggml_tensor * validate_hidden     = ggml_new_tensor_3d(validate_ctx, GGML_TYPE_F32, hidden_dim_, 1, 1);
        ggml_tensor * validate_ids        = ggml_new_tensor_2d(validate_ctx, GGML_TYPE_I32, shortlist_size_, 1);
        ggml_tensor * validate_logits     = ggml_mul_mat_id(validate_ctx, validate_weights_3d, validate_hidden, validate_ids);

        const ggml_backend_dev_t backend_device = ggml_backend_get_device(backend_);
        const bool supported = backend_device != nullptr && ggml_backend_dev_supports_op(backend_device, validate_logits);
        ggml_free(validate_ctx);

        if (emit_debug_logs) {
            LOG_INF("[reduced-lmhead] mul_mat_id direct %s on backend %s\n",
                    supported ? "enabled" : "unsupported",
                    ggml_backend_name(backend_));
        }

        return supported;
    }

    bool upload_opencl_indices_from_host(
            const int32_t * indices,
            int32_t count,
            std::shared_ptr<OpenclI32BufferHandle> & out_handle,
            const char ** source_label,
            std::shared_ptr<OpenclI32BufferHandle> * reusable_upload_indices_device) {
        if (indices == nullptr || count <= 0 || backend_ == nullptr) {
            return false;
        }

        std::shared_ptr<OpenclI32BufferHandle> reusable =
                reusable_upload_indices_device != nullptr ? *reusable_upload_indices_device : opencl_uploaded_indices_device_;

        if (reusable != nullptr &&
            reusable->backend == backend_ &&
            reusable->device_buffer != nullptr &&
            reusable->capacity >= count) {
            if (reduced_lm_head_opencl_device_i32_buffer_write_from_host(
                        backend_,
                        reusable->device_buffer,
                        indices,
                        count)) {
                reusable->count = count;
                opencl_uploaded_indices_device_ = reusable;
                out_handle = reusable;
                if (source_label != nullptr) {
                    *source_label = "host-upload-reuse";
                }
                return true;
            }
            if (reusable_upload_indices_device != nullptr && *reusable_upload_indices_device == reusable) {
                reusable_upload_indices_device->reset();
            }
            if (opencl_uploaded_indices_device_ == reusable) {
                opencl_uploaded_indices_device_.reset();
            }
        }

        void * device_buffer = nullptr;
        if (!reduced_lm_head_opencl_device_i32_buffer_from_host(
                    backend_,
                    indices,
                    count,
                    &device_buffer) ||
            device_buffer == nullptr) {
            return false;
        }

        auto handle = std::make_shared<OpenclI32BufferHandle>();
        handle->backend = backend_;
        handle->device_buffer = device_buffer;
        handle->count = count;
        handle->capacity = count;
        opencl_uploaded_indices_device_ = handle;
        if (reusable_upload_indices_device != nullptr) {
            *reusable_upload_indices_device = handle;
        }
        out_handle = handle;
        if (source_label != nullptr) {
            *source_label = "host-upload";
        }
        return true;
    }

    bool prepare_opencl_indexed_mul_mat(
            bool emit_debug_logs,
            int requested_storage_rows,
            bool /* use_opencl_padded_device_ids */,
            std::shared_ptr<OpenclI32BufferHandle> * reusable_upload_indices_device) {
        storage_shortlist_size_ = shortlist_size_;
        storage_to_output_pos_.clear();
        storage_order_identity_ = true;

        if (backend_ == nullptr || !reduced_lm_head_backend_is_opencl(backend_)) {
            return false;
        }
        if (weight_type_ != GGML_TYPE_Q4_0) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_indexed_mul_mat disabled: LM head type %s is not Q4_0\n",
                        ggml_type_name(weight_type_));
            }
            return false;
        }
        if (source_weights_tensor_ == nullptr || source_weights_tensor_->buffer == nullptr) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_indexed_mul_mat disabled: LM head tensor has no live backend buffer\n");
            }
            return false;
        }
        if (!ggml_is_contiguous(source_weights_tensor_)) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_indexed_mul_mat disabled: LM head tensor is not contiguous\n");
            }
            return false;
        }

        const int requested_rows = requested_storage_rows > 0
                ? std::max(requested_storage_rows, shortlist_size_)
                : shortlist_size_;
        const int padded_rows = opencl_gather_padded_rows(requested_rows);
        if (!reduced_lm_head_opencl_supports_indexed_mul_mat_q4_0(
                    backend_,
                    source_weights_tensor_,
                    padded_rows,
                    8)) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_indexed_mul_mat unsupported on backend %s for logical=%d padded=%d\n",
                        ggml_backend_name(backend_),
                        shortlist_size_,
                        padded_rows);
            }
            return false;
        }

        storage_shortlist_size_ = padded_rows;
        storage_output_indices_.resize(static_cast<size_t>(storage_shortlist_size_));
        std::copy(output_indices_.begin(), output_indices_.end(), storage_output_indices_.begin());
        const int32_t pad_row = storage_output_indices_[shortlist_size_ - 1];
        std::fill(
                storage_output_indices_.begin() + shortlist_size_,
                storage_output_indices_.end(),
                pad_row);

        const bool can_reuse_device_ids =
                opencl_output_indices_device_ != nullptr &&
                opencl_output_indices_device_->device_buffer != nullptr &&
                opencl_output_indices_device_->count >= storage_shortlist_size_;

        opencl_gather_indices_device_.reset();
        opencl_gather_indices_padded_ = false;
        opencl_gather_selected_count_ = 0;
        opencl_gather_pad_row_ = -1;

        const char * ids_source = "host-upload";
        if (can_reuse_device_ids) {
            opencl_gather_indices_device_ = opencl_output_indices_device_;
            ids_source = "selector-device";
        } else {
            upload_opencl_indices_from_host(
                    storage_output_indices_.data(),
                    storage_shortlist_size_,
                    opencl_gather_indices_device_,
                    &ids_source,
                    reusable_upload_indices_device);
        }

        if (opencl_gather_indices_device_ == nullptr ||
            opencl_gather_indices_device_->device_buffer == nullptr) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_indexed_mul_mat disabled: failed to prepare device row ids\n");
            }
            return false;
        }

        if (emit_debug_logs) {
            LOG_INF("[reduced-lmhead] opencl_indexed_mul_mat enabled on backend %s (logical_rows=%d padded_rows=%d ids=%s)\n",
                    ggml_backend_name(backend_),
                    shortlist_size_,
                    storage_shortlist_size_,
                    ids_source);
        }

        return true;
    }

    bool prepare_opencl_gather_mul_mat(
            bool emit_debug_logs,
            int requested_storage_rows,
            bool use_opencl_padded_device_ids,
            std::shared_ptr<OpenclI32BufferHandle> * reusable_upload_indices_device) {
        storage_shortlist_size_ = shortlist_size_;

        if (backend_ == nullptr || !reduced_lm_head_backend_is_opencl(backend_)) {
            return false;
        }
        if (weight_type_ != GGML_TYPE_Q4_0) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_gather_mul_mat disabled: LM head type %s is not Q4_0\n",
                        ggml_type_name(weight_type_));
            }
            return false;
        }
        if (source_weights_tensor_ == nullptr || source_weights_tensor_->buffer == nullptr) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_gather_mul_mat disabled: LM head tensor has no live backend buffer\n");
            }
            return false;
        }
        if (!ggml_is_contiguous(source_weights_tensor_)) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_gather_mul_mat disabled: LM head tensor is not contiguous\n");
            }
            return false;
        }

        const int requested_rows = requested_storage_rows > 0
                ? std::max(requested_storage_rows, shortlist_size_)
                : shortlist_size_;
        const int padded_rows = opencl_gather_padded_rows(requested_rows);
        if (!reduced_lm_head_opencl_supports_gather_rows_q4_0(backend_, source_weights_tensor_, padded_rows)) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] opencl_gather_mul_mat unsupported on backend %s for logical=%d padded=%d\n",
                        ggml_backend_name(backend_),
                        shortlist_size_,
                        padded_rows);
            }
            return false;
        }

        storage_shortlist_size_ = padded_rows;
        storage_output_indices_.resize(static_cast<size_t>(storage_shortlist_size_));

        const bool already_sorted = std::is_sorted(output_indices_.begin(), output_indices_.end());
        if (already_sorted) {
            std::copy(output_indices_.begin(), output_indices_.end(), storage_output_indices_.begin());
            storage_order_identity_ = true;
            storage_to_output_pos_.clear();
        } else {
            storage_order_identity_ = false;
            storage_to_output_pos_.resize(static_cast<size_t>(shortlist_size_));
            std::vector<int32_t> original_to_sorted(static_cast<size_t>(shortlist_size_));
            std::iota(original_to_sorted.begin(), original_to_sorted.end(), 0);
            std::stable_sort(
                    original_to_sorted.begin(),
                    original_to_sorted.end(),
                    [&](int32_t lhs, int32_t rhs) {
                        return output_indices_[lhs] < output_indices_[rhs];
                    });

            for (int sorted_pos = 0; sorted_pos < shortlist_size_; ++sorted_pos) {
                const int32_t original_pos = original_to_sorted[sorted_pos];
                storage_output_indices_[sorted_pos] = output_indices_[original_pos];
                storage_to_output_pos_[sorted_pos] = original_pos;
            }
        }
        const int32_t pad_row = storage_output_indices_[shortlist_size_ - 1];
        std::fill(
                storage_output_indices_.begin() + shortlist_size_,
                storage_output_indices_.end(),
                pad_row);

        const bool can_reuse_device_sorted_ids =
                opencl_output_indices_device_ != nullptr &&
                opencl_output_indices_device_->device_buffer != nullptr &&
                opencl_output_indices_device_->count == storage_shortlist_size_ &&
                already_sorted;
        const bool can_reuse_device_sorted_ids_with_kernel_padding =
                use_opencl_padded_device_ids &&
                opencl_output_indices_device_ != nullptr &&
                opencl_output_indices_device_->device_buffer != nullptr &&
                opencl_output_indices_device_->count == shortlist_size_ &&
                storage_shortlist_size_ >= shortlist_size_ &&
                already_sorted;

        if (emit_debug_logs) {
            LOG_INF("[reduced-lmhead] opencl_gather_mul_mat enabled on backend %s (logical_rows=%d padded_rows=%d sorted_rows=%s)\n",
                    ggml_backend_name(backend_),
                    shortlist_size_,
                    storage_shortlist_size_,
                    can_reuse_device_sorted_ids ? "device-asc" :
                            (can_reuse_device_sorted_ids_with_kernel_padding ? "device-asc+padded" :
                             (already_sorted ? "no-op" : "host-upload")));
        }

        opencl_gather_indices_device_.reset();
        opencl_gather_indices_padded_ = false;
        opencl_gather_selected_count_ = 0;
        opencl_gather_pad_row_ = -1;
        if (can_reuse_device_sorted_ids) {
            opencl_gather_indices_device_ = opencl_output_indices_device_;
            return true;
        }
        if (can_reuse_device_sorted_ids_with_kernel_padding) {
            opencl_gather_indices_device_ = opencl_output_indices_device_;
            opencl_gather_indices_padded_ = storage_shortlist_size_ > shortlist_size_;
            opencl_gather_selected_count_ = shortlist_size_;
            opencl_gather_pad_row_ = pad_row;
            return true;
        }

        const char * ids_source = nullptr;
        if (!upload_opencl_indices_from_host(
                    storage_output_indices_.data(),
                    storage_shortlist_size_,
                    opencl_gather_indices_device_,
                    &ids_source,
                    reusable_upload_indices_device) &&
            emit_debug_logs) {
            LOG_INF("[reduced-lmhead] failed to pre-upload sorted gather ids; falling back to host-upload gather path\n");
        }

        return true;
    }

    bool can_use_gather_mul_mat(bool emit_debug_logs) const {
        if (source_weights_tensor_ == nullptr || source_weights_tensor_->buffer == nullptr) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] gather_mul_mat disabled: LM head tensor has no live backend buffer\n");
            }
            return false;
        }
        if (!ggml_is_contiguous(source_weights_tensor_)) {
            if (emit_debug_logs) {
                LOG_INF("[reduced-lmhead] gather_mul_mat disabled: LM head tensor is not contiguous\n");
            }
            return false;
        }

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(8, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * validate_ctx = ggml_init(params);
        if (validate_ctx == nullptr) {
            return false;
        }

        ggml_tensor * validate_ids     = ggml_new_tensor_1d(validate_ctx, GGML_TYPE_I32, shortlist_size_);
        ggml_tensor * validate_gathered = ggml_get_rows(validate_ctx, source_weights_tensor_, validate_ids);
        ggml_tensor * validate_hidden  = ggml_new_tensor_2d(validate_ctx, GGML_TYPE_F32, hidden_dim_, 1);
        ggml_tensor * validate_logits  = ggml_mul_mat(validate_ctx, validate_gathered, validate_hidden);

        const ggml_backend_dev_t backend_device = ggml_backend_get_device(backend_);
        const bool get_rows_ok = backend_device != nullptr && ggml_backend_dev_supports_op(backend_device, validate_gathered);
        const bool mul_mat_ok  = backend_device != nullptr && ggml_backend_dev_supports_op(backend_device, validate_logits);
        ggml_free(validate_ctx);

        const bool supported = get_rows_ok && mul_mat_ok;
        if (emit_debug_logs) {
            LOG_INF("[reduced-lmhead] gather_mul_mat %s on backend %s (get_rows=%s, mul_mat=%s)\n",
                    supported ? "enabled" : "unsupported",
                    ggml_backend_name(backend_),
                    get_rows_ok ? "yes" : "no",
                    mul_mat_ok  ? "yes" : "no");
        }
        return supported;
    }

    std::vector<int32_t> build_ids_tensor_values(int batch_size) const {
        std::vector<int32_t> ids(static_cast<size_t>(shortlist_size_) * batch_size);
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            std::memcpy(
                    ids.data() + static_cast<size_t>(batch_idx) * shortlist_size_,
                    output_indices_.data(),
                    sizeof(int32_t) * shortlist_size_);
        }
        return ids;
    }

    // Max vocabulary rows per gather_mul_mat graph step (F32 dequant buffer = chunk * hidden_dim * 4 bytes).
    static constexpr int kGatherMulMatChunkLen = 2048;

    void reorder_storage_logits_to_output_order(
            const float * storage_logits,
            int batch_size,
            std::vector<float> & logits_out) const {
        GGML_ASSERT(storage_logits != nullptr);
        GGML_ASSERT(storage_order_identity_ || storage_to_output_pos_.size() == static_cast<size_t>(shortlist_size_));
        GGML_ASSERT(logits_out.size() == static_cast<size_t>(shortlist_size_) * batch_size);

        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const float * src = storage_logits + static_cast<size_t>(batch_idx) * storage_shortlist_size_;
            float * dst = logits_out.data() + static_cast<size_t>(batch_idx) * shortlist_size_;
            if (storage_order_identity_) {
                std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(shortlist_size_));
            } else {
                for (int storage_pos = 0; storage_pos < shortlist_size_; ++storage_pos) {
                    dst[storage_to_output_pos_[storage_pos]] = src[storage_pos];
                }
            }
        }
    }

    static uint64_t variant_map_key(int batch_size, int sublist_graph_len) {
        return (uint64_t(uint32_t(batch_size)) << 32) | uint32_t(sublist_graph_len);
    }

    GraphVariant * get_or_create_variant(int batch_size, int sublist_graph_len, std::string & error) {
        const int sl_key = mode_ == Mode::GATHER_MUL_MAT ? sublist_graph_len : storage_shortlist_size_;
        const uint64_t key = variant_map_key(batch_size, sl_key);
        auto it = variants_.find(key);
        if (it != variants_.end()) {
            return it->second.get();
        }

        auto variant = std::make_unique<GraphVariant>();

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(8, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        variant->ctx = ggml_init(params);
        if (variant->ctx == nullptr) {
            error = "ggml_init failed for reduced LM head batch variant";
            return nullptr;
        }

        if (mode_ == Mode::DIRECT_MUL_MAT_ID) {
            if (source_weights_tensor_ == nullptr) {
                error = "mul_mat_id direct mode is missing source weights tensor";
                return nullptr;
            }

            variant->weights_tensor = ggml_reshape_3d(variant->ctx, source_weights_tensor_, hidden_dim_, 1, source_vocab_out_);
            variant->hidden_tensor  = ggml_new_tensor_3d(variant->ctx, GGML_TYPE_F32, hidden_dim_, 1, batch_size);
            variant->ids_tensor     = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_I32, shortlist_size_, batch_size);
            variant->logits_tensor  = ggml_mul_mat_id(variant->ctx, variant->weights_tensor, variant->hidden_tensor, variant->ids_tensor);

            ggml_set_input(variant->hidden_tensor);
            ggml_set_input(variant->ids_tensor);
            ggml_set_output(variant->logits_tensor);
        } else if (mode_ == Mode::GATHER_MUL_MAT) {
            if (source_weights_tensor_ == nullptr) {
                error = "gather_mul_mat mode is missing source weights tensor";
                return nullptr;
            }
            if (sublist_graph_len <= 0 || sublist_graph_len > shortlist_size_) {
                error = "invalid gather sublist length for reduced LM head variant";
                return nullptr;
            }

            variant->ids_tensor     = ggml_new_tensor_1d(variant->ctx, GGML_TYPE_I32, sublist_graph_len);
            variant->gathered_tensor = ggml_get_rows(variant->ctx, source_weights_tensor_, variant->ids_tensor);
            variant->hidden_tensor  = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_F32, hidden_dim_, batch_size);
            variant->logits_tensor  = ggml_mul_mat(variant->ctx, variant->gathered_tensor, variant->hidden_tensor);

            ggml_set_input(variant->ids_tensor);
            ggml_set_input(variant->hidden_tensor);
            ggml_set_output(variant->logits_tensor);
        } else if (mode_ == Mode::OPENCL_INDEXED_MUL_MAT) {
            variant->hidden_tensor = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_F32, hidden_dim_, batch_size);
            variant->logits_tensor = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_F32, storage_shortlist_size_, batch_size);

            ggml_set_input(variant->hidden_tensor);
            ggml_set_output(variant->logits_tensor);
        } else {
            if (mode_ == Mode::OPENCL_GATHER_MUL_MAT) {
                if (!ensure_opencl_gather_weights_ready(error)) {
                    return nullptr;
                }
                variant->weights_tensor = shared_weights_tensor_;
            } else {
                variant->weights_tensor = ggml_new_tensor_2d(variant->ctx, weight_type_, hidden_dim_, storage_shortlist_size_);
            }
            variant->hidden_tensor  = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_F32, hidden_dim_, batch_size);
            variant->logits_tensor  = ggml_mul_mat(variant->ctx, variant->weights_tensor, variant->hidden_tensor);

            ggml_set_input(variant->hidden_tensor);
            ggml_set_output(variant->logits_tensor);
        }

        if (mode_ != Mode::OPENCL_INDEXED_MUL_MAT) {
            variant->graph = ggml_new_graph_custom(variant->ctx, 8, false);
            ggml_build_forward_expand(variant->graph, variant->logits_tensor);
        }

        if ((mode_ == Mode::OPENCL_GATHER_MUL_MAT || mode_ == Mode::OPENCL_INDEXED_MUL_MAT) && emit_debug_logs_) {
            LOG_INF("[reduced-lmhead] creating OpenCL matmul variant batch=%d rows=%d (mode=%s)\n",
                    batch_size,
                    storage_shortlist_size_,
                    mode_name());
        }

        variant->buffer = ggml_backend_alloc_ctx_tensors(variant->ctx, backend_);
        if (variant->buffer == nullptr) {
            error = "failed to allocate reduced LM head compute buffer";
            return nullptr;
        }
        ggml_backend_buffer_set_usage(variant->buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        if ((mode_ == Mode::OPENCL_GATHER_MUL_MAT || mode_ == Mode::OPENCL_INDEXED_MUL_MAT) && emit_debug_logs_) {
            LOG_INF("[reduced-lmhead] OpenCL matmul variant buffer allocated\n");
        }

        if (mode_ == Mode::DIRECT_MUL_MAT_ID) {
            const std::vector<int32_t> ids = build_ids_tensor_values(batch_size);
            ggml_backend_tensor_set(variant->ids_tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
        } else if (mode_ == Mode::GATHER_MUL_MAT) {
            // Indices are uploaded per chunk in compute_logits_batch.
        } else if (mode_ == Mode::OPENCL_GATHER_MUL_MAT) {
            // Shared gathered weights were already materialized once in
            // ensure_opencl_gather_weights_ready() and are reused across all
            // batch-shape variants for this shortlist.
        } else if (mode_ == Mode::OPENCL_INDEXED_MUL_MAT) {
            // The source weights and device row ids are passed directly to the
            // tuned OpenCL kernel at compute time; no packed tensor is stored here.
        } else {
            ggml_backend_tensor_set(variant->weights_tensor, packed_weights_.data(), 0, packed_weights_.size());
        }

        GraphVariant * variant_ptr = variant.get();
        variants_.emplace(key, std::move(variant));
        return variant_ptr;
    }

    ggml_backend_t backend_ = nullptr; // borrowed
    ggml_type weight_type_ = GGML_TYPE_F32;
    ggml_tensor * source_weights_tensor_ = nullptr; // borrowed when direct mode is enabled
    std::shared_ptr<OpenclI32BufferHandle> opencl_output_indices_device_;
    std::shared_ptr<OpenclI32BufferHandle> opencl_gather_indices_device_;
    std::shared_ptr<OpenclI32BufferHandle> opencl_uploaded_indices_device_;
    bool opencl_gather_indices_padded_ = false;
    int opencl_gather_selected_count_ = 0;
    int32_t opencl_gather_pad_row_ = -1;
    std::vector<uint8_t> packed_weights_;
    std::vector<int32_t> output_indices_;
    std::vector<int32_t> storage_output_indices_;
    std::vector<int32_t> storage_to_output_pos_;
    bool storage_order_identity_ = true;
    std::vector<float> storage_logits_scratch_;
    std::unordered_map<uint64_t, std::unique_ptr<GraphVariant>> variants_;
    Mode mode_ = Mode::PACKED_MUL_MAT;
    int hidden_dim_ = 0;
    int shortlist_size_ = 0;
    int storage_shortlist_size_ = 0;
    int source_vocab_out_ = 0;
    bool emit_debug_logs_ = false;
    ggml_context * shared_weights_ctx_ = nullptr;
    ggml_backend_buffer_t shared_weights_buffer_ = nullptr;
    ggml_tensor * shared_weights_tensor_ = nullptr;
    bool shared_weights_enqueued_ = false;
    bool shared_weights_ready_ = false;
};

static std::vector<llama_token_data> build_reduced_candidate_logits(
        const std::vector<llama_token> & token_ids,
        const std::vector<float> & logits) {
    std::vector<llama_token_data> candidates;
    if (token_ids.empty() || token_ids.size() != logits.size()) {
        return candidates;
    }

    candidates.reserve(token_ids.size());
    for (size_t i = 0; i < token_ids.size(); ++i) {
        candidates.push_back(llama_token_data{token_ids[i], logits[i], 0.0f});
    }

    return candidates;
}

static std::vector<llama_token_data> apply_reduced_sampler(
        struct common_sampler * sampler,
        const std::vector<llama_token> & token_ids,
        const std::vector<float> & logits) {
    if (sampler == nullptr) {
        return {};
    }
    if (token_ids.empty() || token_ids.size() != logits.size()) {
        return {};
    }

    llama_token_data_array * sampled_candidates = common_sampler_apply_logits(
            sampler,
            token_ids.data(),
            logits.data(),
            token_ids.size(),
            true,
            true);
    if (sampled_candidates == nullptr || sampled_candidates->data == nullptr || sampled_candidates->size == 0) {
        return {};
    }

    return std::vector<llama_token_data>(
            sampled_candidates->data,
            sampled_candidates->data + sampled_candidates->size);
}

static void debug_compare_reduced_logits_with_full(
        const RoundSelection & round_selection,
        const std::vector<float> & reduced_logits,
        llama_context * ctx_dft,
        int batch_idx,
        const char * tag) {
    if (ctx_dft == nullptr) {
        return;
    }
    if (round_selection.output_indices.size() != reduced_logits.size()) {
        return;
    }

    llama_synchronize(ctx_dft);
    const float * full_logits = llama_get_logits_ith(ctx_dft, batch_idx);
    if (full_logits == nullptr) {
        LOG_INF("[%s] full logits unavailable for batch_idx=%d\n", tag, batch_idx);
        return;
    }

    std::vector<float> ref_logits(reduced_logits.size(), 0.0f);
    float max_abs_diff = 0.0f;
    double sum_abs_diff = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < reduced_logits.size(); ++i) {
        ref_logits[i] = full_logits[round_selection.output_indices[i]];
        const float abs_diff = std::fabs(ref_logits[i] - reduced_logits[i]);
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            worst_idx = i;
        }
        sum_abs_diff += abs_diff;
    }

    LOG_INF("[%s] reduced-vs-full shortlist logits: size=%zu max_abs_diff=%.6f mean_abs_diff=%.6f worst_token=%d output_idx=%d reduced=%.6f full=%.6f\n",
            tag,
            reduced_logits.size(),
            max_abs_diff,
            reduced_logits.empty() ? 0.0 : static_cast<float>(sum_abs_diff / reduced_logits.size()),
            reduced_logits.empty() ? -1 : round_selection.token_ids[worst_idx],
            reduced_logits.empty() ? -1 : round_selection.output_indices[worst_idx],
            reduced_logits.empty() ? 0.0f : reduced_logits[worst_idx],
            reduced_logits.empty() ? 0.0f : ref_logits[worst_idx]);

    std::vector<llama_token_data> reduced_top = build_reduced_candidate_logits(round_selection.token_ids, reduced_logits);
    std::vector<llama_token_data> ref_top = build_reduced_candidate_logits(round_selection.token_ids, ref_logits);
    auto by_logit_desc = [](const llama_token_data & a, const llama_token_data & b) {
        if (a.logit != b.logit) {
            return a.logit > b.logit;
        }
        return a.id < b.id;
    };
    std::sort(reduced_top.begin(), reduced_top.end(), by_logit_desc);
    std::sort(ref_top.begin(), ref_top.end(), by_logit_desc);

    print_candidate_distribution("reduced-raw-top", ctx_dft, reduced_top);
    print_candidate_distribution("full-shortlist-top", ctx_dft, ref_top);
}

static bool ensure_debug_trimmed_round_selection(
        const RoundSelection & current_round_selection,
        DebugTrimmedLmHeadReference & debug_ref,
        bool emit_debug_logs,
        std::string & error) {
    if (!debug_ref.enabled) {
        error = "trimmed GGUF reference is disabled";
        return false;
    }
    if (debug_ref.cached_round_selection.projector != nullptr &&
        debug_ref.cached_token_ids == current_round_selection.token_ids) {
        return true;
    }

    RoundSelection trimmed_round_selection;
    trimmed_round_selection.token_ids = current_round_selection.token_ids;
    trimmed_round_selection.selector_scores = current_round_selection.selector_scores;
    trimmed_round_selection.output_indices.reserve(current_round_selection.token_ids.size());

    for (llama_token token_id : current_round_selection.token_ids) {
        auto it = debug_ref.lm_head_ctx.token_to_output_idx.find(token_id);
        if (it == debug_ref.lm_head_ctx.token_to_output_idx.end()) {
            std::ostringstream oss;
            oss << "trimmed GGUF is missing token id " << token_id << " from the current shortlist";
            error = oss.str();
            return false;
        }
        trimmed_round_selection.output_indices.push_back(it->second);
    }

    trimmed_round_selection.projector = std::make_shared<ReducedLmHeadProjector>();
    if (!trimmed_round_selection.projector->initialize(
                debug_ref.lm_head_ctx,
                trimmed_round_selection.output_indices,
                nullptr,
                emit_debug_logs,
                error)) {
        return false;
    }

    debug_ref.cached_token_ids = current_round_selection.token_ids;
    debug_ref.cached_round_selection = std::move(trimmed_round_selection);
    debug_ref.logged_weight_compare_for_cache = false;
    return true;
}

static void debug_compare_reduced_logits_with_trimmed(
        const RoundSelection & current_round_selection,
        const std::vector<float> & reduced_logits,
        const float * hidden,
        DebugTrimmedLmHeadReference & debug_ref,
        const llama_context * ctx_dft,
        const char * tag,
        bool emit_debug_logs) {
    if (!debug_ref.enabled || hidden == nullptr) {
        return;
    }
    if (current_round_selection.projector == nullptr) {
        return;
    }
    if (current_round_selection.output_indices.size() != reduced_logits.size()) {
        return;
    }

    std::string error;
    if (!ensure_debug_trimmed_round_selection(current_round_selection, debug_ref, emit_debug_logs, error)) {
        LOG_INF("[%s] trimmed GGUF compare unavailable: %s\n", tag, error.c_str());
        return;
    }

    const auto & dynamic_weights = current_round_selection.projector->packed_weights();
    const auto & trimmed_weights = debug_ref.cached_round_selection.projector->packed_weights();
    if (!debug_ref.logged_weight_compare_for_cache) {
        if (dynamic_weights.empty() || trimmed_weights.empty()) {
            LOG_INF("[%s] packed-weight-compare skipped: dynamic_mode=%s trimmed_mode=%s\n",
                    tag,
                    current_round_selection.projector->mode_name(),
                    debug_ref.cached_round_selection.projector->mode_name());
        } else if (dynamic_weights.size() != trimmed_weights.size()) {
            LOG_INF("[%s] packed-weight-compare: size mismatch dynamic=%zu trimmed=%zu\n",
                    tag, dynamic_weights.size(), trimmed_weights.size());
        } else {
            size_t mismatch_count = 0;
            size_t first_mismatch = dynamic_weights.size();
            for (size_t i = 0; i < dynamic_weights.size(); ++i) {
                if (dynamic_weights[i] != trimmed_weights[i]) {
                    ++mismatch_count;
                    if (first_mismatch == dynamic_weights.size()) {
                        first_mismatch = i;
                    }
                }
            }

            if (mismatch_count == 0) {
                LOG_INF("[%s] packed-weight-compare: identical (%zu bytes, type=%s)\n",
                        tag,
                        dynamic_weights.size(),
                        ggml_type_name(current_round_selection.projector->weight_type()));
            } else {
                LOG_INF("[%s] packed-weight-compare: mismatched bytes=%zu/%zu first_mismatch=%zu dynamic=%u trimmed=%u\n",
                        tag,
                        mismatch_count,
                        dynamic_weights.size(),
                        first_mismatch,
                        static_cast<unsigned>(dynamic_weights[first_mismatch]),
                        static_cast<unsigned>(trimmed_weights[first_mismatch]));
            }
        }
        debug_ref.logged_weight_compare_for_cache = true;
    }

    std::vector<float> trimmed_logits;
    if (!debug_ref.cached_round_selection.projector->compute_logits(hidden, trimmed_logits, nullptr, error)) {
        LOG_INF("[%s] trimmed GGUF reduced projection failed: %s\n", tag, error.c_str());
        return;
    }
    if (trimmed_logits.size() != reduced_logits.size()) {
        LOG_INF("[%s] trimmed GGUF reduced projection returned size=%zu expected=%zu\n",
                tag, trimmed_logits.size(), reduced_logits.size());
        return;
    }

    float max_abs_diff = 0.0f;
    double sum_abs_diff = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < reduced_logits.size(); ++i) {
        const float abs_diff = std::fabs(trimmed_logits[i] - reduced_logits[i]);
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            worst_idx = i;
        }
        sum_abs_diff += abs_diff;
    }

    LOG_INF("[%s] dynamic-vs-trimmed logits: size=%zu max_abs_diff=%.6f mean_abs_diff=%.6f worst_token=%d dynamic=%.6f trimmed=%.6f\n",
            tag,
            reduced_logits.size(),
            max_abs_diff,
            reduced_logits.empty() ? 0.0f : static_cast<float>(sum_abs_diff / reduced_logits.size()),
            reduced_logits.empty() ? -1 : current_round_selection.token_ids[worst_idx],
            reduced_logits.empty() ? 0.0f : reduced_logits[worst_idx],
            trimmed_logits.empty() ? 0.0f : trimmed_logits[worst_idx]);

    std::vector<llama_token_data> dynamic_top = build_reduced_candidate_logits(current_round_selection.token_ids, reduced_logits);
    std::vector<llama_token_data> trimmed_top = build_reduced_candidate_logits(current_round_selection.token_ids, trimmed_logits);
    auto by_logit_desc = [](const llama_token_data & a, const llama_token_data & b) {
        if (a.logit != b.logit) {
            return a.logit > b.logit;
        }
        return a.id < b.id;
    };
    std::sort(dynamic_top.begin(), dynamic_top.end(), by_logit_desc);
    std::sort(trimmed_top.begin(), trimmed_top.end(), by_logit_desc);

    print_candidate_distribution("dynamic-reduced-top", ctx_dft, dynamic_top);
    print_candidate_distribution("trimmed-gguf-top", ctx_dft, trimmed_top);
}

static llama_token selected_token_from_candidates(const llama_token_data_array * candidates) {
    if (candidates == nullptr || candidates->data == nullptr || candidates->size == 0) {
        return LLAMA_TOKEN_NULL;
    }
    if (candidates->selected < 0 || static_cast<size_t>(candidates->selected) >= candidates->size) {
        return LLAMA_TOKEN_NULL;
    }

    return candidates->data[candidates->selected].id;
}

static void debug_compare_reduced_batch_with_trimmed(
        const RoundSelection & current_round_selection,
        const std::vector<int> & active_seq_ids,
        const std::vector<struct common_sampler *> & dynamic_samplers,
        const std::vector<struct common_sampler *> & reference_samplers,
        const std::vector<float> & hidden_batch,
        const std::vector<std::vector<float>> & dynamic_raw_logits,
        DebugTrimmedLmHeadReference & debug_ref,
        int depth,
        bool emit_debug_logs) {
    if (!debug_ref.enabled) {
        return;
    }
    if (active_seq_ids.empty() || hidden_batch.empty() || dynamic_raw_logits.empty()) {
        return;
    }
    if (dynamic_samplers.size() != active_seq_ids.size() ||
        reference_samplers.size() != active_seq_ids.size() ||
        dynamic_raw_logits.size() != active_seq_ids.size()) {
        return;
    }

    std::string error;
    if (!ensure_debug_trimmed_round_selection(current_round_selection, debug_ref, emit_debug_logs, error)) {
        LOG_INF("[trimmed-depth-compare] depth=%d unavailable: %s\n", depth, error.c_str());
        return;
    }

    std::vector<float> trimmed_batched_logits;
    if (!debug_ref.cached_round_selection.projector->compute_logits_batch(
                hidden_batch.data(),
                static_cast<int>(active_seq_ids.size()),
                trimmed_batched_logits,
                nullptr,
                error)) {
        LOG_INF("[trimmed-depth-compare] depth=%d projection failed: %s\n", depth, error.c_str());
        return;
    }

    const size_t shortlist_size = current_round_selection.token_ids.size();
    if (shortlist_size == 0) {
        return;
    }
    if (trimmed_batched_logits.size() != shortlist_size * active_seq_ids.size()) {
        LOG_INF("[trimmed-depth-compare] depth=%d returned size=%zu expected=%zu\n",
                depth,
                trimmed_batched_logits.size(),
                shortlist_size * active_seq_ids.size());
        return;
    }

    int top1_mismatch_count = 0;
    int selected_mismatch_count = 0;
    double total_mean_abs_diff = 0.0;
    float global_max_abs_diff = 0.0f;
    int worst_seq_id = -1;
    llama_token worst_token_id = LLAMA_TOKEN_NULL;
    llama_token first_dynamic_top1 = LLAMA_TOKEN_NULL;
    llama_token first_trimmed_top1 = LLAMA_TOKEN_NULL;
    llama_token first_dynamic_selected = LLAMA_TOKEN_NULL;
    llama_token first_trimmed_selected = LLAMA_TOKEN_NULL;
    bool logged_first_detail = false;

    for (size_t active_idx = 0; active_idx < active_seq_ids.size(); ++active_idx) {
        const auto & dynamic_logits = dynamic_raw_logits[active_idx];
        if (dynamic_logits.size() != shortlist_size) {
            continue;
        }

        const float * trimmed_logits = trimmed_batched_logits.data() + active_idx * shortlist_size;
        float seq_max_abs_diff = 0.0f;
        double seq_sum_abs_diff = 0.0;
        size_t dynamic_best_idx = 0;
        size_t trimmed_best_idx = 0;

        for (size_t token_idx = 0; token_idx < shortlist_size; ++token_idx) {
            if (dynamic_logits[token_idx] > dynamic_logits[dynamic_best_idx]) {
                dynamic_best_idx = token_idx;
            }
            if (trimmed_logits[token_idx] > trimmed_logits[trimmed_best_idx]) {
                trimmed_best_idx = token_idx;
            }

            const float abs_diff = std::fabs(dynamic_logits[token_idx] - trimmed_logits[token_idx]);
            seq_max_abs_diff = std::max(seq_max_abs_diff, abs_diff);
            seq_sum_abs_diff += abs_diff;
            if (abs_diff > global_max_abs_diff) {
                global_max_abs_diff = abs_diff;
                worst_seq_id = active_seq_ids[active_idx];
                worst_token_id = current_round_selection.token_ids[token_idx];
            }
        }

        const float seq_mean_abs_diff = static_cast<float>(seq_sum_abs_diff / shortlist_size);
        total_mean_abs_diff += seq_mean_abs_diff;

        const llama_token dynamic_top1 = current_round_selection.token_ids[dynamic_best_idx];
        const llama_token trimmed_top1 = current_round_selection.token_ids[trimmed_best_idx];
        if (dynamic_top1 != trimmed_top1) {
            ++top1_mismatch_count;
        }

        const llama_token dynamic_selected = selected_token_from_candidates(
                common_sampler_get_candidates(dynamic_samplers[active_idx], true));
        llama_token trimmed_selected = LLAMA_TOKEN_NULL;
        if (reference_samplers[active_idx] != nullptr) {
            const llama_token_data_array * reference_candidates = common_sampler_apply_logits(
                    reference_samplers[active_idx],
                    current_round_selection.token_ids.data(),
                    trimmed_logits,
                    shortlist_size,
                    true,
                    true);
            trimmed_selected = selected_token_from_candidates(reference_candidates);
        }
        if (dynamic_selected != trimmed_selected) {
            ++selected_mismatch_count;
        }

        if (!logged_first_detail &&
            (dynamic_top1 != trimmed_top1 || dynamic_selected != trimmed_selected || seq_max_abs_diff > 1e-3f)) {
            logged_first_detail = true;
            first_dynamic_top1 = dynamic_top1;
            first_trimmed_top1 = trimmed_top1;
            first_dynamic_selected = dynamic_selected;
            first_trimmed_selected = trimmed_selected;
            LOG_INF("[trimmed-depth-compare] depth=%d seq=%d max_abs_diff=%.6f mean_abs_diff=%.6f top1_dynamic=%d top1_trimmed=%d selected_dynamic=%d selected_trimmed=%d\n",
                    depth,
                    active_seq_ids[active_idx],
                    seq_max_abs_diff,
                    seq_mean_abs_diff,
                    dynamic_top1,
                    trimmed_top1,
                    dynamic_selected,
                    trimmed_selected);
        }
    }

    LOG_INF("[trimmed-depth-summary] depth=%d seqs=%zu top1_mismatch=%d selected_mismatch=%d worst_seq=%d worst_token=%d max_abs_diff=%.6f mean_abs_diff=%.6f\n",
            depth,
            active_seq_ids.size(),
            top1_mismatch_count,
            selected_mismatch_count,
            worst_seq_id,
            worst_token_id,
            global_max_abs_diff,
            active_seq_ids.empty() ? 0.0f : static_cast<float>(total_mean_abs_diff / active_seq_ids.size()));

    if (!logged_first_detail && (top1_mismatch_count > 0 || selected_mismatch_count > 0)) {
        LOG_INF("[trimmed-depth-summary] depth=%d first-detail top1_dynamic=%d top1_trimmed=%d selected_dynamic=%d selected_trimmed=%d\n",
                depth,
                first_dynamic_top1,
                first_trimmed_top1,
                first_dynamic_selected,
                first_trimmed_selected);
    }
}

static std::vector<llama_token_data> compute_candidates_from_projector(
        const float * hidden,
        RoundSelection & round_selection,
        struct common_sampler * sampler,
        std::vector<float> * raw_logits_out = nullptr) {
    if (round_selection.projector == nullptr) {
        return {};
    }

    std::vector<float> logits;
    std::string error;
    if (!round_selection.projector->compute_logits(hidden, logits, nullptr, error)) {
        fprintf(stderr, "[reduced-lmhead] branch reduced projection failed: %s\n", error.c_str());
        return {};
    }

    if (raw_logits_out != nullptr) {
        *raw_logits_out = logits;
    }

    return apply_reduced_sampler(sampler, round_selection.token_ids, logits);
}

static bool compute_candidates_from_projector_batch(
        const float * hidden_batch,
        int batch_size,
        RoundSelection & round_selection,
        const std::vector<struct common_sampler *> & samplers,
        std::vector<std::vector<float>> * raw_logits_out,
        ReducedDraftSamplingProfile * profile,
        std::string & error) {
    if (round_selection.projector == nullptr) {
        error = "projector is not initialized";
        return false;
    }
    if (batch_size <= 0) {
        error = "batch_size must be positive";
        return false;
    }
    if (samplers.size() != static_cast<size_t>(batch_size)) {
        error = "sampler count does not match batch size";
        return false;
    }

    std::vector<float> batched_logits;
    const auto logits_compute_start = ggml_time_us();
    if (!round_selection.projector->compute_logits_batch(hidden_batch, batch_size, batched_logits, nullptr, error)) {
        return false;
    }
    const auto logits_compute_end = ggml_time_us();
    if (profile != nullptr) {
        profile->logits_compute_us += (logits_compute_end - logits_compute_start);
    }

    const size_t shortlist_size = round_selection.token_ids.size();
    if (raw_logits_out != nullptr) {
        raw_logits_out->assign(batch_size, {});
    }

    const auto sampler_apply_start = ggml_time_us();
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const size_t offset = shortlist_size * batch_idx;
        const float * logits = batched_logits.data() + offset;
        if (raw_logits_out != nullptr) {
            (*raw_logits_out)[batch_idx].assign(logits, logits + shortlist_size);
        }
        llama_token_data_array * sampled_candidates = common_sampler_apply_logits(
                samplers[batch_idx],
                round_selection.token_ids.data(),
                logits,
                shortlist_size,
                true,
                true);
        if (sampled_candidates == nullptr || sampled_candidates->data == nullptr || sampled_candidates->size == 0) {
            error = "common_sampler_apply_logits returned no candidates";
            return false;
        }
    }
    const auto sampler_apply_end = ggml_time_us();
    if (profile != nullptr) {
        profile->sampler_apply_us += (sampler_apply_end - sampler_apply_start);
    }

    return true;
}

static bool compute_candidates_from_fused_ctx_logits_batch(
        llama_context * ctx_dft,
        RoundSelection & round_selection,
        const std::vector<int> & batch_indices,
        const std::vector<struct common_sampler *> & samplers,
        std::vector<std::vector<float>> * raw_logits_out,
        ReducedDraftSamplingProfile * profile,
        std::string & error) {
    if (ctx_dft == nullptr) {
        error = "draft context is null";
        return false;
    }
    if (batch_indices.empty()) {
        error = "batch_indices is empty";
        return false;
    }
    if (batch_indices.size() != samplers.size()) {
        error = "sampler count does not match batch index count";
        return false;
    }

    const size_t shortlist_size = round_selection.token_ids.size();
    if (shortlist_size == 0) {
        error = "shortlist is empty";
        return false;
    }

    if (raw_logits_out != nullptr) {
        raw_logits_out->assign(batch_indices.size(), {});
    }

    std::vector<std::vector<float>> fetched_logits(batch_indices.size());
    const auto logits_fetch_start = ggml_time_us();
    for (size_t i = 0; i < batch_indices.size(); ++i) {
        const float * full_logits = llama_get_logits_ith(ctx_dft, batch_indices[i]);
        if (full_logits == nullptr) {
            std::ostringstream oss;
            oss << "llama_get_logits_ith returned null for batch_idx=" << batch_indices[i];
            error = oss.str();
            return false;
        }
        fetched_logits[i].assign(full_logits, full_logits + shortlist_size);
        if (raw_logits_out != nullptr) {
            (*raw_logits_out)[i] = fetched_logits[i];
        }
    }
    const auto logits_fetch_end = ggml_time_us();
    if (profile != nullptr) {
        profile->logits_fetch_us += (logits_fetch_end - logits_fetch_start);
    }

    const auto sampler_apply_start = ggml_time_us();
    for (size_t i = 0; i < batch_indices.size(); ++i) {
        llama_token_data_array * sampled_candidates = common_sampler_apply_logits(
                samplers[i],
                round_selection.token_ids.data(),
                fetched_logits[i].data(),
                shortlist_size,
                true,
                true);
        if (sampled_candidates == nullptr || sampled_candidates->data == nullptr || sampled_candidates->size == 0) {
            error = "common_sampler_apply_logits returned no candidates";
            return false;
        }
    }
    const auto sampler_apply_end = ggml_time_us();
    if (profile != nullptr) {
        profile->sampler_apply_us += (sampler_apply_end - sampler_apply_start);
    }

    return true;
}

static RoundSelection build_round_selection_from_selector_result(
        const SelectorResult & selector_result,
        const ReducedLmHeadContext & lm_head_ctx,
        bool dump_selector_scores,
        bool dump_reduced_logits,
        bool emit_debug_logs,
        const llama_context * ctx_dft,
        std::shared_ptr<ReducedLmHeadProjector> reusable_projector = nullptr,
        RoundSelectionProfile * profile = nullptr,
        bool force_packed = false,
        bool prepare_runtime_output = false,
        int runtime_output_rows = 0,
        const DynamicSelectorConfig * selector_config = nullptr,
        RuntimeRowBucketState * runtime_bucket_state = nullptr,
        std::unordered_map<int, std::shared_ptr<ReducedLmHeadProjector>> * projector_cache = nullptr,
        std::shared_ptr<OpenclI32BufferHandle> * reusable_upload_indices_device = nullptr) {
    (void) dump_reduced_logits;

    RoundSelection round_selection;
    if (selector_result.token_ids.empty() && selector_result.output_indices.empty()) {
        return round_selection;
    }

    const auto shortlist_filter_start = ggml_time_us();
    if (!selector_result.output_indices.empty() && lm_head_ctx.output_idx_matches_token_id) {
        round_selection.token_ids.reserve(selector_result.output_indices.size());
        if (dump_selector_scores) {
            round_selection.selector_scores.reserve(selector_result.output_indices.size());
        }
        round_selection.output_indices.reserve(selector_result.output_indices.size());

        for (size_t i = 0; i < selector_result.output_indices.size(); ++i) {
            const int32_t output_idx = selector_result.output_indices[i];
            if (output_idx < 0 || output_idx >= lm_head_ctx.vocab_out) {
                ++round_selection.dropped_token_count;
                continue;
            }

            round_selection.output_indices.push_back(output_idx);
            round_selection.token_ids.push_back(static_cast<llama_token>(output_idx));
            if (dump_selector_scores) {
                round_selection.selector_scores.push_back(i < selector_result.scores.size() ? selector_result.scores[i] : 0.0f);
            }
        }
        round_selection.opencl_output_indices_device = selector_result.opencl_output_indices_device;
    } else {
        std::unordered_set<llama_token> seen_tokens;
        seen_tokens.reserve(selector_result.token_ids.size());
        round_selection.token_ids.reserve(selector_result.token_ids.size());
        if (dump_selector_scores) {
            round_selection.selector_scores.reserve(selector_result.scores.size());
        }
        round_selection.output_indices.reserve(selector_result.token_ids.size());

        for (size_t i = 0; i < selector_result.token_ids.size(); ++i) {
            const llama_token token_id = selector_result.token_ids[i];
            if (!seen_tokens.insert(token_id).second) {
                continue;
            }

            auto it = lm_head_ctx.token_to_output_idx.find(token_id);
            if (it == lm_head_ctx.token_to_output_idx.end()) {
                ++round_selection.dropped_token_count;
                continue;
            }

            round_selection.token_ids.push_back(token_id);
            if (dump_selector_scores) {
                round_selection.selector_scores.push_back(i < selector_result.scores.size() ? selector_result.scores[i] : 0.0f);
            }
            round_selection.output_indices.push_back(it->second);
        }
    }
    const auto shortlist_filter_end = ggml_time_us();
    if (profile != nullptr) {
        profile->shortlist_filter_us += (shortlist_filter_end - shortlist_filter_start);
    }

    if (emit_debug_logs && dump_selector_scores) {
        print_token_scores("selector-shortlist", ctx_dft, round_selection.token_ids, round_selection.selector_scores);
        if (round_selection.dropped_token_count > 0) {
            LOG_INF("[selector-shortlist] dropped %d token ids that are missing from the draft output vocab\n",
                    round_selection.dropped_token_count);
        }
    }

    if (round_selection.token_ids.empty()) {
        fprintf(stderr, "[reduced-lmhead] selector shortlist is empty after LM head vocab filtering\n");
        return RoundSelection{};
    }

    const bool prepare_indexed_graph_output =
            !prepare_runtime_output &&
            selector_config != nullptr &&
            selector_config->opencl_indexed_lmhead &&
            selector_config->opencl_indexed_lmhead_in_graph;

    int requested_runtime_output_rows = runtime_output_rows;
    if ((prepare_runtime_output || prepare_indexed_graph_output) &&
        selector_config != nullptr &&
        selector_config->runtime_bucket_enabled) {
        const int bucket_rows = choose_runtime_bucket_rows(
                static_cast<int>(round_selection.output_indices.size()),
                lm_head_ctx.vocab_out,
                *selector_config,
                runtime_bucket_state);
        if (bucket_rows > 0) {
            requested_runtime_output_rows = bucket_rows;
        }
    }

    // TODO(packed-fallback cleanup): retire packed row gathering once direct mul_mat_id also covers trimmed-GGUF sources.
    // TODO(selector+lmhead fusion): explore fusing selector projection and reduced LM head once shortlist semantics settle.
    if (emit_debug_logs) {
        LOG_INF("[reduced-lmhead] preparing reduced quantized LM head for %zu selected rows (runtime_rows=%d type=%s, hidden_dim=%d)\n",
                round_selection.output_indices.size(),
                requested_runtime_output_rows,
                ggml_type_name(lm_head_ctx.tensor->type),
                lm_head_ctx.hidden_dim);
    }

    const auto projector_init_start = ggml_time_us();
    const int projector_cache_key = (prepare_runtime_output || prepare_indexed_graph_output) && requested_runtime_output_rows > 0
            ? requested_runtime_output_rows
            : 0;
    const bool use_projector_cache =
            projector_cache != nullptr &&
            selector_config != nullptr &&
            selector_config->projector_cache_limit > 0 &&
            projector_cache_key > 0;
    if (use_projector_cache) {
        auto cached_it = projector_cache->find(projector_cache_key);
        if (cached_it == projector_cache->end()) {
            while (static_cast<int>(projector_cache->size()) >= selector_config->projector_cache_limit &&
                   !projector_cache->empty()) {
                projector_cache->erase(projector_cache->begin());
            }
            cached_it = projector_cache->emplace(projector_cache_key, std::make_shared<ReducedLmHeadProjector>()).first;
        }
        auto & cached_projector = cached_it->second;
        if (cached_projector == nullptr) {
            cached_projector = std::make_shared<ReducedLmHeadProjector>();
        }
        round_selection.projector = cached_projector;
    } else {
        round_selection.projector = reusable_projector != nullptr
                ? std::move(reusable_projector)
                : std::make_shared<ReducedLmHeadProjector>();
    }
    std::string projector_error;
    if (!round_selection.projector->initialize(
                lm_head_ctx,
                round_selection.output_indices,
                round_selection.opencl_output_indices_device,
                emit_debug_logs,
                projector_error,
                force_packed,
                false,
                requested_runtime_output_rows,
                selector_config != nullptr && selector_config->opencl_padded_device_ids,
                selector_config == nullptr || selector_config->opencl_indexed_lmhead,
                reusable_upload_indices_device)) {
        fprintf(stderr, "[reduced-lmhead] failed to initialize reduced projector: %s\n", projector_error.c_str());
        return RoundSelection{};
    }

    if (prepare_runtime_output) {
        std::string runtime_projector_error;
        const int effective_runtime_output_rows = requested_runtime_output_rows > 0
                ? requested_runtime_output_rows
                : round_selection.projector->runtime_output_row_capacity();
        if (effective_runtime_output_rows <= 0 ||
            effective_runtime_output_rows < static_cast<int>(round_selection.output_indices.size())) {
            fprintf(stderr, "[reduced-lmhead] invalid runtime output rows: rows=%d shortlist=%zu\n",
                    effective_runtime_output_rows,
                    round_selection.output_indices.size());
            return RoundSelection{};
        }

        ggml_tensor * runtime_output_source_tensor = nullptr;
        if (!round_selection.projector->runtime_output_copy_source_tensor(
                    effective_runtime_output_rows,
                    &runtime_output_source_tensor,
                    runtime_projector_error)) {
            fprintf(stderr, "[reduced-lmhead] failed to prepare runtime output copy source: %s\n",
                    runtime_projector_error.c_str());
            return RoundSelection{};
        }
        if (runtime_output_source_tensor != nullptr) {
            round_selection.runtime_output_rows = effective_runtime_output_rows;
            round_selection.runtime_output_source_tensor = runtime_output_source_tensor;
            round_selection.runtime_output_borrowed = true;
        } else {
        // NOTE: a zero-copy borrowed-tensor path was tested here, but on the
        // current OpenCL EAGLE draft path it can lead to use-after-free /
        // scheduler lifetime issues. Keep the safer explicit packed export +
        // upload path until backend-owned tensor lifetime is handled robustly.
        std::vector<uint8_t> host_packed_weights;
        if (!round_selection.projector->export_packed_weights(host_packed_weights, runtime_projector_error)) {
            fprintf(stderr, "[reduced-lmhead] failed to export runtime packed shortlist: %s\n",
                    runtime_projector_error.c_str());
            return RoundSelection{};
        }
        const std::vector<uint8_t> * packed_src = &host_packed_weights;

        const size_t row_bytes = lm_head_ctx.tensor->nb[1];
        const size_t shortlist_size = round_selection.output_indices.size();
        if (packed_src->size() != row_bytes * shortlist_size) {
            fprintf(stderr, "[reduced-lmhead] packed shortlist size mismatch: got=%zu expected=%zu\n",
                    packed_src->size(),
                    row_bytes * shortlist_size);
            return RoundSelection{};
        }

        round_selection.runtime_output_rows = effective_runtime_output_rows;
        round_selection.runtime_output_weights.resize(row_bytes * static_cast<size_t>(effective_runtime_output_rows));
        std::memcpy(
                round_selection.runtime_output_weights.data(),
                packed_src->data(),
                packed_src->size());

        const uint8_t * pad_row = packed_src->data() + row_bytes * (shortlist_size - 1);
        for (int row = static_cast<int>(shortlist_size); row < effective_runtime_output_rows; ++row) {
            std::memcpy(
                    round_selection.runtime_output_weights.data() + row_bytes * static_cast<size_t>(row),
                    pad_row,
                    row_bytes);
        }
        }
    }
    const auto projector_init_end = ggml_time_us();
    if (profile != nullptr) {
        profile->projector_init_us += (projector_init_end - projector_init_start);
    }
    if (emit_debug_logs) {
        LOG_INF("[reduced-lmhead] prepared reduced projector in %.2f ms (mode=%s)\n",
                (projector_init_end - projector_init_start) / 1000.0,
                round_selection.projector->mode_name());
    }

    return round_selection;
}

static RoundSelection run_selector_then_reduced_lm_head(
        CandidateSelector & selector,
        const ReducedLmHeadContext & lm_head_ctx,
        const float * hidden,
        int hidden_dim,
        int selector_top_k,
        bool dump_selector_scores,
        bool dump_reduced_logits,
        bool emit_debug_logs,
        const llama_context * ctx_dft,
        std::shared_ptr<ReducedLmHeadProjector> reusable_projector = nullptr,
        RoundSelectionProfile * profile = nullptr,
        bool force_packed = false,
        bool prepare_runtime_output = false,
        int runtime_output_rows = 0,
        const DynamicSelectorConfig * selector_config = nullptr,
        RuntimeRowBucketState * runtime_bucket_state = nullptr,
        std::unordered_map<int, std::shared_ptr<ReducedLmHeadProjector>> * projector_cache = nullptr,
        std::shared_ptr<OpenclI32BufferHandle> * reusable_upload_indices_device = nullptr) {
    const SelectorResult selector_result = selector.run(
            hidden,
            hidden_dim,
            selector_top_k,
            profile != nullptr ? &profile->selector : nullptr,
            dump_selector_scores);
    if (selector_result.token_ids.empty() && selector_result.output_indices.empty()) {
        fprintf(stderr, "[selector] selector '%s' returned no token ids\n", selector.name());
        return {};
    }

    return build_round_selection_from_selector_result(
            selector_result,
            lm_head_ctx,
            dump_selector_scores,
            dump_reduced_logits,
            emit_debug_logs,
            ctx_dft,
            std::move(reusable_projector),
            profile,
            force_packed,
            prepare_runtime_output,
            runtime_output_rows,
            selector_config,
            runtime_bucket_state,
            projector_cache,
            reusable_upload_indices_device);
}

} // namespace


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

static bool build_target_shortlist_candidates(
        llama_context * ctx_tgt,
        int sampled_tgt_batch_idx,
        const std::vector<llama_token> & shortlist_token_ids,
        std::vector<llama_token_data> & out_candidates,
        std::string & error) {
    if (ctx_tgt == nullptr) {
        error = "null target context";
        return false;
    }
    if (sampled_tgt_batch_idx < 0) {
        error = "negative target batch index";
        return false;
    }
    if (shortlist_token_ids.empty()) {
        error = "target shortlist is empty";
        return false;
    }

    const float * full_logits = llama_get_logits_ith(ctx_tgt, sampled_tgt_batch_idx);
    if (full_logits == nullptr) {
        std::ostringstream oss;
        oss << "failed to read full target logits row " << sampled_tgt_batch_idx;
        error = oss.str();
        return false;
    }

    out_candidates.resize(shortlist_token_ids.size());
    for (size_t i = 0; i < shortlist_token_ids.size(); ++i) {
        const llama_token token_id = shortlist_token_ids[i];
        if (token_id < 0) {
            std::ostringstream oss;
            oss << "invalid shortlist token id " << token_id << " at index " << i;
            error = oss.str();
            return false;
        }
        out_candidates[i] = llama_token_data{ token_id, full_logits[token_id], 0.0f };
    }

    return true;
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
    DynamicSelectorConfig selector_config;
    std::string debug_compare_trimmed_gguf_path;

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
        } else if (arg == "--selector-top-k" && i + 1 < argc) {
            selector_config.top_k = std::stoi(argv[++i]);
        } else if ((arg == "--selector-softmax-threshold" || arg == "--selector-top-p") && i + 1 < argc) {
            selector_config.selector_softmax_threshold = std::stof(argv[++i]);
            selector_config.selector_softmax_threshold_enabled = true;
        } else if (arg == "--selector-runtime-buckets" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "off" || value == "0" || value == "false") {
                selector_config.runtime_bucket_enabled = false;
                selector_config.runtime_buckets.clear();
            } else {
                selector_config.runtime_bucket_enabled = true;
                if (value == "auto" || value == "default") {
                    selector_config.runtime_buckets.clear();
                } else {
                    selector_config.runtime_buckets = parse_positive_int_list(value);
                }
            }
        } else if (arg == "--selector-runtime-bucket-shrink-ratio" && i + 1 < argc) {
            selector_config.runtime_bucket_shrink_ratio = std::stof(argv[++i]);
            selector_config.runtime_bucket_enabled = true;
        } else if (arg == "--selector-runtime-bucket-shrink-patience" && i + 1 < argc) {
            selector_config.runtime_bucket_shrink_patience = std::stoi(argv[++i]);
            selector_config.runtime_bucket_enabled = true;
        } else if ((arg == "--selector-projector-cache-size" || arg == "--selector-projector-cache-limit") && i + 1 < argc) {
            selector_config.projector_cache_limit = std::stoi(argv[++i]);
        } else if (arg == "--selector-hardcoded-ids" && i + 1 < argc) {
            ++i;
            fprintf(stderr, "--selector-hardcoded-ids was removed; this example now supports only the QNN selector path\n");
            return 1;
        } else if (arg == "--selector-hardcoded-ids-file" && i + 1 < argc) {
            ++i;
            fprintf(stderr, "--selector-hardcoded-ids-file was removed; this example now supports only the QNN selector path\n");
            return 1;
        } else if (arg == "--reduced-lmhead-gguf" && i + 1 < argc) {
            selector_config.reduced_lmhead_gguf = argv[++i];
        } else if (arg == "--selector-ctx-dir" && i + 1 < argc) {
            selector_config.selector_ctx_dir = argv[++i];
        } else if (arg == "--selector-json" && i + 1 < argc) {
            selector_config.selector_json_path = argv[++i];
        } else if (arg == "--selector-bin" && i + 1 < argc) {
            selector_config.selector_bin_path = argv[++i];
        } else if (arg == "--selector-backend-so" && i + 1 < argc) {
            selector_config.selector_backend_so = argv[++i];
        } else if (arg == "--selector-system-so" && i + 1 < argc) {
            selector_config.selector_system_so = argv[++i];
        } else if (arg == "--selector-hot-vocab-json" && i + 1 < argc) {
            selector_config.selector_hot_vocab_json = argv[++i];
        } else if (arg == "--dump-selector-scores") {
            selector_config.dump_selector_scores = true;
        } else if (arg == "--dump-reduced-logits") {
            selector_config.dump_reduced_logits = true;
        } else if (arg == "--use-reduced-lmhead") {
            selector_config.use_reduced_lmhead = true;
        } else if (arg == "--force-packed-mul-mat") {
            selector_config.force_packed_mul_mat = true;
        } else if (arg == "--selector-launch-after-recompute") {
            selector_config.selector_launch_after_recompute = true;
        } else if (arg == "--selector-launch-before-recompute") {
            selector_config.selector_launch_after_recompute = false;
        } else if (arg == "--selector-opencl-padded-device-ids") {
            selector_config.opencl_padded_device_ids = true;
        } else if (arg == "--selector-opencl-indexed-lmhead") {
            selector_config.opencl_indexed_lmhead = true;
        } else if (arg == "--selector-disable-opencl-indexed-lmhead") {
            selector_config.opencl_indexed_lmhead = false;
        } else if (arg == "--selector-opencl-indexed-lmhead-in-graph") {
            selector_config.opencl_indexed_lmhead_in_graph = true;
        } else if (arg == "--selector-opencl-indexed-lmhead-external") {
            selector_config.opencl_indexed_lmhead_in_graph = false;
        } else if (arg == "--selector-cpu-softmax-threshold") {
            selector_config.selector_force_cpu_softmax_threshold = true;
            selector_config.selector_force_opencl_softmax_threshold = false;
        } else if (arg == "--selector-opencl-softmax-threshold") {
            selector_config.selector_force_opencl_softmax_threshold = true;
            selector_config.selector_force_cpu_softmax_threshold = false;
        } else if (arg == "--debug-compare-trimmed-gguf" && i + 1 < argc) {
            debug_compare_trimmed_gguf_path = argv[++i];
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
            printf("Dynamic Selector / Reduced LM Head Options:\n");
            printf("  --selector-top-k N                      Shortlist size/cap for selector output; 0 means uncapped with --selector-top-p (default: 64)\n");
            printf("  --selector-top-p P                      Select ids with softmax(score) >= P; --selector-top-k is the optional cap\n");
            printf("  --selector-softmax-threshold P          Alias for --selector-top-p\n");
            printf("  --selector-runtime-buckets CSV|auto|off Bucket runtime LMHead rows to reduce realloc/init churn (default: off)\n");
            printf("  --selector-runtime-bucket-shrink-ratio R Shrink bucket after selected_rows <= current*R (default: 0.5)\n");
            printf("  --selector-runtime-bucket-shrink-patience N Rounds before shrinking bucket (default: 8)\n");
            printf("  --selector-projector-cache-size N       Runtime bucket projector cache entries; 0 disables cache (default: 1)\n");
            printf("  --reduced-lmhead-gguf FILE              Use LM head weights from a trimmed GGUF instead of gathering from the full draft GGUF\n");
            printf("  --selector-ctx-dir PATH                 QNN selector artifact dir or forward_0_json.json path\n");
            printf("  --selector-json FILE                    Explicit selector QNN JSON path (overrides --selector-ctx-dir)\n");
            printf("  --selector-bin FILE                     Explicit selector QNN context binary path\n");
            printf("  --selector-backend-so FNAME             Override QNN backend .so for selector graph\n");
            printf("  --selector-system-so FNAME              Override QNN system .so for selector graph\n");
            printf("  --selector-hot-vocab-json FILE          Diagnostic: report selector hot/cold coverage against token-id JSON array\n");
            printf("  --dump-selector-scores                  Print selector shortlist for each round\n");
            printf("  --dump-reduced-logits                   Print reduced LM head logits for each round\n");
            printf("  --use-reduced-lmhead                    Use shortlist gather + reduced LM head for draft expansion\n");
            printf("  --force-packed-mul-mat                  Force packed_mul_mat mode (skip mul_mat_id even if supported)\n\n");
            printf("  --selector-launch-after-recompute       Launch async selector after hidden-only recompute to avoid GPU contention\n");
            printf("  --selector-launch-before-recompute      Launch async selector before recompute (default)\n\n");
            printf("  --selector-opencl-padded-device-ids     Experimental: reuse device ids with kernel-side runtime row padding\n\n");
            printf("  --selector-opencl-indexed-lmhead-in-graph  Use graph-integrated indexed LMHead after root depth (experimental)\n");
            printf("  --selector-opencl-indexed-lmhead-external  Use external indexed LMHead for every draft depth (default)\n\n");
            printf("  --selector-cpu-softmax-threshold        Select softmax-threshold ids on CPU, then upload ids only\n");
            printf("  --selector-opencl-softmax-threshold     Force OpenCL softmax-threshold helper\n\n");
            printf("  --debug-compare-trimmed-gguf FILE       Compare dynamic reduced logits against a trimmed EAGLE GGUF at root depth\n\n");
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

    if (selector_config.top_k < 0) {
        LOG_ERR("%s: --selector-top-k must be >= 0\n", __func__);
        return 1;
    }
    if (!selector_config.selector_softmax_threshold_enabled && selector_config.top_k <= 0) {
        LOG_ERR("%s: --selector-top-k must be > 0 unless --selector-top-p/--selector-softmax-threshold is enabled\n", __func__);
        return 1;
    }
    if (selector_config.selector_softmax_threshold_enabled &&
        (!std::isfinite(selector_config.selector_softmax_threshold) ||
         selector_config.selector_softmax_threshold <= 0.0f ||
         selector_config.selector_softmax_threshold > 1.0f)) {
        LOG_ERR("%s: --selector-top-p/--selector-softmax-threshold must be in (0, 1]\n", __func__);
        return 1;
    }
    if (selector_config.runtime_bucket_enabled) {
        if (selector_config.runtime_bucket_shrink_ratio <= 0.0f ||
            selector_config.runtime_bucket_shrink_ratio >= 1.0f ||
            !std::isfinite(selector_config.runtime_bucket_shrink_ratio)) {
            LOG_ERR("%s: --selector-runtime-bucket-shrink-ratio must be in (0, 1)\n", __func__);
            return 1;
        }
        if (selector_config.runtime_bucket_shrink_patience < 1) {
            LOG_ERR("%s: --selector-runtime-bucket-shrink-patience must be >= 1\n", __func__);
            return 1;
        }
    }
    if (selector_config.projector_cache_limit < 0) {
        LOG_ERR("%s: --selector-projector-cache-size/--selector-projector-cache-limit must be >= 0\n", __func__);
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
    selector_config.debug_log_level = params.qnn_log_level;

    if (selector_config.selector_backend_so.empty()) {
        selector_config.selector_backend_so = qnn_config.backend_so;
    }
    if (selector_config.selector_system_so.empty()) {
        selector_config.selector_system_so = qnn_config.system_so;
    }
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

    llama_set_eagle_hidden_only(ctx_dft, selector_config.use_reduced_lmhead);

    const int hidden_dim = llama_model_n_embd(model_dft);
    if (hidden_dim != kExpectedHiddenDim) {
        LOG_ERR("%s: expected selector hidden dim %d but draft model reports %d\n",
                __func__, kExpectedHiddenDim, hidden_dim);
        return 1;
    }

    ReducedLmHeadContext lm_head_ctx;
    if (!prepare_lm_head_context(model_dft, ctx_dft, params.speculative.model.path, lm_head_ctx)) {
        LOG_ERR("%s: failed to prepare reduced LM head context\n", __func__);
        return 1;
    }

    if (lm_head_ctx.hidden_dim != hidden_dim) {
        LOG_ERR("%s: LM head hidden dim mismatch: lm_head=%d hidden=%d\n",
                __func__, lm_head_ctx.hidden_dim, hidden_dim);
        return 1;
    }

    DebugTrimmedLmHeadReference debug_trimmed_ref;
    DebugTrimmedLmHeadReference trimmed_lm_head_ref;
    if (!selector_config.reduced_lmhead_gguf.empty()) {
        if (!selector_config.use_reduced_lmhead) {
            LOG_ERR("%s: --reduced-lmhead-gguf requires --use-reduced-lmhead\n", __func__);
            return 1;
        }
        if (!file_exists_readable(selector_config.reduced_lmhead_gguf)) {
            LOG_ERR("%s: reduced LM head GGUF is not readable: %s\n",
                    __func__, selector_config.reduced_lmhead_gguf.c_str());
            return 1;
        }

        std::string trimmed_lm_head_error;
        if (!prepare_debug_trimmed_lm_head_reference(
                    selector_config.reduced_lmhead_gguf,
                    model_dft->arch_name(),
                    lm_head_ctx,
                    trimmed_lm_head_ref,
                    trimmed_lm_head_error)) {
            LOG_ERR("%s: failed to prepare reduced LM head GGUF source: %s\n",
                    __func__, trimmed_lm_head_error.c_str());
            return 1;
        }

        lm_head_ctx = trimmed_lm_head_ref.lm_head_ctx;
        LOG_INF("[reduced-lmhead] using trimmed GGUF LM head source: '%s' (output_vocab=%d)\n",
                selector_config.reduced_lmhead_gguf.c_str(),
                lm_head_ctx.vocab_out);
    }
    if (!debug_compare_trimmed_gguf_path.empty()) {
        if (!selector_config.use_reduced_lmhead) {
            LOG_ERR("%s: --debug-compare-trimmed-gguf requires --use-reduced-lmhead\n", __func__);
            return 1;
        }
        if (!file_exists_readable(debug_compare_trimmed_gguf_path)) {
            LOG_ERR("%s: debug compare trimmed GGUF is not readable: %s\n",
                    __func__, debug_compare_trimmed_gguf_path.c_str());
            return 1;
        }

        std::string debug_compare_error;
        if (!prepare_debug_trimmed_lm_head_reference(
                    debug_compare_trimmed_gguf_path,
                    model_dft->arch_name(),
                    lm_head_ctx,
                    debug_trimmed_ref,
                    debug_compare_error)) {
            LOG_ERR("%s: failed to prepare trimmed GGUF compare reference: %s\n",
                    __func__, debug_compare_error.c_str());
            return 1;
        }

        LOG_INF("[reduced-lmhead] debug compare against trimmed GGUF enabled: '%s' (output_vocab=%d)\n",
                debug_compare_trimmed_gguf_path.c_str(),
                debug_trimmed_ref.lm_head_ctx.vocab_out);
    }

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

    std::vector<uint8_t> selector_hot_vocab_mask;
    int64_t selector_hot_vocab_unique_ids = 0;
    if (!selector_config.selector_hot_vocab_json.empty()) {
        std::vector<llama_token> hot_token_ids;
        std::string hot_vocab_error;
        if (!load_token_id_json_array(
                    selector_config.selector_hot_vocab_json,
                    hot_token_ids,
                    hot_vocab_error)) {
            LOG_ERR("%s: failed to load --selector-hot-vocab-json: %s\n",
                    __func__, hot_vocab_error.c_str());
            return 1;
        }

        const int vocab_token_count = llama_vocab_n_tokens(vocab_dft);
        selector_hot_vocab_mask.assign(static_cast<size_t>(std::max(0, vocab_token_count)), 0);
        int64_t ignored_ids = 0;
        for (llama_token token_id : hot_token_ids) {
            if (token_id < 0 || token_id >= vocab_token_count) {
                ++ignored_ids;
                continue;
            }
            if (selector_hot_vocab_mask[static_cast<size_t>(token_id)] == 0) {
                selector_hot_vocab_mask[static_cast<size_t>(token_id)] = 1;
                ++selector_hot_vocab_unique_ids;
            }
        }
        if (selector_hot_vocab_unique_ids <= 0) {
            LOG_ERR("%s: --selector-hot-vocab-json contained no token ids valid for draft vocab size %d\n",
                    __func__, vocab_token_count);
            return 1;
        }
        LOG_INF("[selector-hot-vocab] coverage diagnostics enabled: '%s' unique_ids=%lld ignored=%lld vocab=%d\n",
                selector_config.selector_hot_vocab_json.c_str(),
                (long long) selector_hot_vocab_unique_ids,
                (long long) ignored_ids,
                vocab_token_count);
    }

    const bool selector_softmax_uncapped =
            selector_config.selector_softmax_threshold_enabled &&
            selector_config.top_k == 0;
    const int selector_output_limit = selector_softmax_uncapped
            ? lm_head_ctx.vocab_out
            : selector_config.top_k;

    std::unique_ptr<CandidateSelector> candidate_selector;
    int64_t selector_predecode_init_us = 0;
    if (selector_config.use_reduced_lmhead) {
        if (selector_output_limit <= 0) {
            LOG_ERR("%s: selector output limit is invalid: %d\n", __func__, selector_output_limit);
            return 1;
        }

        std::string selector_json_path;
        std::string selector_bin_path;
        if (!resolve_selector_artifact_paths(selector_config, selector_json_path, selector_bin_path)) {
            LOG_ERR("%s: reduced LM head selector requires --selector-ctx-dir, --selector-json, or both --selector-json/--selector-bin\n",
                    __func__);
            return 1;
        }
        LOG_INF("[selector-qnn] using json='%s' bin='%s'\n",
                selector_json_path.c_str(),
                selector_bin_path.c_str());

        candidate_selector = build_candidate_selector(
                selector_config,
                lm_head_ctx.backend,
                lm_head_ctx.output_idx_matches_token_id,
                lm_head_ctx.output_idx_matches_token_id &&
                lm_head_ctx.tensor != nullptr &&
                lm_head_ctx.tensor->type == GGML_TYPE_Q4_0 &&
                lm_head_ctx.backend != nullptr &&
                reduced_lm_head_backend_is_opencl(lm_head_ctx.backend) &&
                !selector_config.opencl_indexed_lmhead &&
                selector_output_limit >= 512 &&
                (selector_output_limit % 8) == 0);
        if (candidate_selector == nullptr) {
            LOG_ERR("%s: failed to build QNN candidate selector\n", __func__);
            return 1;
        }
        {
            std::string warmup_error;
            const int64_t warmup_start_us = ggml_time_us();
            if (!candidate_selector->warmup(&warmup_error)) {
                LOG_ERR("%s: failed to initialize selector before decode: %s\n",
                        __func__, warmup_error.c_str());
                return 1;
            }
            selector_predecode_init_us = ggml_time_us() - warmup_start_us;
            LOG_INF("[selector-qnn] selector context preloaded before decode in %.3f ms\n",
                    selector_predecode_init_us / 1000.0);
        }
        LOG_INF("[reduced-lmhead] enabled with QNN selector selector_limit=%d output_vocab=%d hidden_dim=%d\n",
                selector_output_limit,
                lm_head_ctx.vocab_out,
                hidden_dim);
        if (selector_config.selector_softmax_threshold_enabled) {
            if (selector_softmax_uncapped) {
                LOG_INF("[selector-qnn] selector softmax threshold enabled: p>=%.8g max_selected=uncapped(output_vocab=%d) runtime_rows=selected/padded\n",
                        static_cast<double>(selector_config.selector_softmax_threshold),
                        lm_head_ctx.vocab_out);
            } else {
                LOG_INF("[selector-qnn] selector softmax threshold enabled: p>=%.8g max_selected=%d runtime_rows=selected/padded\n",
                        static_cast<double>(selector_config.selector_softmax_threshold),
                        selector_output_limit);
            }
            const bool prefer_cpu_softmax_threshold =
                    selector_config.selector_force_cpu_softmax_threshold ||
                    (selector_config.opencl_indexed_lmhead && !selector_config.selector_force_opencl_softmax_threshold);
            LOG_INF("[selector-qnn] selector softmax threshold backend: %s%s\n",
                    prefer_cpu_softmax_threshold ? "cpu-host" : "opencl-helper",
                    prefer_cpu_softmax_threshold ? " (ids uploaded to OpenCL LMHead)" : "");
        }
        if (selector_config.runtime_bucket_enabled) {
            const std::vector<int> runtime_buckets = selector_config.runtime_buckets.empty()
                    ? default_runtime_buckets(lm_head_ctx.vocab_out)
                    : selector_config.runtime_buckets;
            LOG_INF("[selector-qnn] runtime row buckets enabled: [%s] shrink_ratio=%.3f patience=%d\n",
                    join_int_list(runtime_buckets).c_str(),
                    static_cast<double>(selector_config.runtime_bucket_shrink_ratio),
                    selector_config.runtime_bucket_shrink_patience);
        }
        LOG_INF("[selector-qnn] runtime projector cache size limit: %d%s\n",
                selector_config.projector_cache_limit,
                selector_config.projector_cache_limit == 0 ? " (disabled)" : "");
        LOG_INF("[selector-qnn] OpenCL padded device ids: %s\n",
                selector_config.opencl_padded_device_ids ? "enabled" : "disabled");
        LOG_INF("[selector-qnn] OpenCL indexed reduced LMHead: %s%s\n",
                selector_config.opencl_indexed_lmhead ? "enabled" : "disabled",
                selector_config.opencl_indexed_lmhead
                        ? (selector_config.opencl_indexed_lmhead_in_graph
                                ? " (root external, tree graph-integrated)"
                                : " (external-all-depths)")
                        : "");
        LOG_INF("[selector-qnn] async selector launch timing: %s\n",
                selector_config.selector_launch_after_recompute ? "after-recompute" : "before-recompute");
        LOG_INF("[reduced-lmhead] target verification stays on full target logits to match eagle-2-qnn vocab-trim behavior\n");
    }

    LOG_INF("[draft-path] mode=%s\n",
            selector_config.use_reduced_lmhead ? "reduced-lmhead" : "baseline-trimmed/full");

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
    if (final_hs.size() % hidden_dim != 0) {
        LOG_ERR("%s: invalid final_hiddens size %zu for hidden_dim=%d\n", __func__, final_hs.size(), hidden_dim);
        return 1;
    }

    // sliced_data: hidden states for tokens[0..n_input-2] (EAGLE draft uses shifted input)
    std::vector<float> sliced_data(final_hs.begin(), final_hs.begin() + (n_input - 1) * hidden_dim);
    // backup_data: all n_input hidden states from prefill
    // i_batch_tgt[0] = n_input-1 indexes the last prompt token's hidden state
    std::vector<float> backup_data = final_hs;

    // Draft model prefill with EAGLE hidden state sharing
    cb_data.data.clear();
    llama_decode_eagle(ctx_dft, llama_batch_get_one(inp.data() + 1, n_input - 1), sliced_data.data());
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
    RoundSelection current_round_selection;
    bool current_round_selection_future_pending = false;
    uint64_t current_round_selection_job_id = 0;
    uint64_t next_round_selection_job_id = 0;
    auto reusable_dynamic_projector = std::make_shared<ReducedLmHeadProjector>();
    std::shared_ptr<OpenclI32BufferHandle> reusable_indexed_id_buffer;
    std::unordered_map<int, std::shared_ptr<ReducedLmHeadProjector>> runtime_projector_cache;
    RuntimeRowBucketState runtime_bucket_state;
    RoundSelectionWorkerState current_round_selection_worker_state;
    std::thread current_round_selection_worker;
    RoundSelectionWorkerGuard current_round_selection_worker_guard {
        &current_round_selection_worker_state,
        &current_round_selection_worker,
    };

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
    int64_t total_draft_forward_submit_us = 0;
    int64_t total_draft_forward_sync_us = 0;
    int32_t total_draft_forward_calls = 0;
    int64_t total_draft_recompute_state_reset_us = 0;
    int64_t total_draft_recompute_replay_decode_us = 0;
    int64_t total_draft_recompute_current_decode_us = 0;
    int64_t total_draft_runtime_output_upload_us = 0;
    int64_t total_draft_runtime_output_borrowed = 0;
    int64_t total_draft_runtime_output_copied = 0;
    int64_t total_draft_recompute_steps = 0;
    int64_t total_draft_recompute_replay_steps = 0;
    int64_t total_main_selector_total_us = 0;
    int64_t total_main_selector_run_us = 0;
    int64_t total_main_selector_init_us = 0;
    int64_t total_main_selector_input_write_us = 0;
    int64_t total_main_selector_graph_execute_us = 0;
    int64_t total_main_selector_output_read_us = 0;
    int64_t total_main_selector_topk_us = 0;
    int64_t total_main_selector_shortlist_filter_us = 0;
    int64_t total_main_selector_projector_init_us = 0;
    int64_t total_main_selector_wait_stall_us = 0;
    int64_t total_main_selector_selected_rows = 0;
    int64_t total_main_selector_runtime_rows = 0;
    int64_t max_main_selector_selected_rows = 0;
    int64_t max_main_selector_runtime_rows = 0;
    int64_t total_main_selector_hot_vocab_rows = 0;
    int64_t total_main_selector_cold_vocab_rows = 0;
    int64_t max_main_selector_cold_vocab_rows = 0;
    int64_t total_main_selector_runtime_row_switches = 0;
    int32_t prev_main_selector_runtime_rows = -1;
    int64_t total_main_selector_window_to_lmhead_us = 0;
    int64_t total_main_selector_exposed_after_lmhead_us = 0;
    int64_t max_main_selector_exposed_after_lmhead_us = 0;
    int64_t total_main_selector_launch_submit_us = 0;
    int64_t total_main_selector_launch_to_dequeue_us = 0;
    int64_t total_main_selector_dequeue_to_start_us = 0;
    int64_t total_main_selector_launch_to_start_us = 0;
    int64_t total_main_selector_launch_to_end_us = 0;
    int64_t total_main_selector_task_start_to_lmhead_us = 0;
    int64_t max_main_selector_launch_to_start_us = 0;
    int64_t max_main_selector_launch_to_end_us = 0;
    int main_selector_rounds_launched = 0;
    int main_selector_rounds_completed = 0;
    int main_selector_rounds_with_lmhead_window = 0;
    int main_selector_rounds_hidden_before_lmhead = 0;
    int64_t current_round_selection_launch_us = 0;
    int64_t current_round_selection_task_start_us = 0;
    int64_t current_round_selection_task_end_us = 0;
    bool current_round_selection_lmhead_window_recorded = false;
    
    // New fine-grained variables for Tree Expansion
    int64_t total_expansion_sampling_us = 0;
    int64_t total_reduced_draft_logits_compute_us = 0;
    int64_t total_reduced_fused_logits_fetch_us = 0;
    int64_t total_reduced_draft_sampler_apply_us = 0;
    int64_t total_reduced_fused_sampler_apply_us = 0;
    int64_t total_expansion_loop_prep_us = 0;
    int64_t total_expansion_selector_prep_us = 0;
    int64_t total_expansion_selector_postprocess_us = 0;
    int64_t total_expansion_candidate_fetch_us = 0;
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
    int64_t total_reduced_target_shortlist_build_us = 0;
    int64_t total_reduced_target_sampler_apply_us = 0;
    int64_t total_reduced_target_accept_piece_us = 0;
    int64_t total_reduced_target_apply_calls = 0;
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

	    if (selector_config.use_reduced_lmhead) {
	        current_round_selection_worker = std::thread([&]() {
	            for (;;) {
	                std::vector<float> hidden_copy;
	                uint64_t job_id = 0;
                    int64_t job_launch_us = 0;
                    int64_t worker_dequeue_us = 0;

	                {
	                    std::unique_lock<std::mutex> lock(current_round_selection_worker_state.mutex);
	                    current_round_selection_worker_state.cv.wait(lock, [&]() {
	                        return current_round_selection_worker_state.stop ||
                               current_round_selection_worker_state.has_pending_job;
                    });

	                    if (current_round_selection_worker_state.stop &&
	                        !current_round_selection_worker_state.has_pending_job) {
	                        break;
	                    }

                        worker_dequeue_us = ggml_time_us();
	                    hidden_copy = std::move(current_round_selection_worker_state.hidden_input);
	                    job_id = current_round_selection_worker_state.pending_job_id;
                        job_launch_us = current_round_selection_worker_state.pending_launch_us;
	                    current_round_selection_worker_state.has_pending_job = false;
	                    current_round_selection_worker_state.has_ready_result = false;
	                    current_round_selection_worker_state.error = nullptr;
	                }

                RoundSelectionTaskResult task_result;
	                std::exception_ptr task_error;

	                try {
                        task_result.profile.launch_us = job_launch_us;
                        task_result.profile.worker_dequeue_us = worker_dequeue_us;
	                    task_result.profile.task_start_us = ggml_time_us();
	                    task_result.selection = run_selector_then_reduced_lm_head(
	                            *candidate_selector,
                            lm_head_ctx,
                            hidden_copy.data(),
                            hidden_dim,
                            selector_output_limit,
                            selector_config.dump_selector_scores,
                            selector_config.dump_reduced_logits,
                            selector_config.debug_log_level >= 2,
                            ctx_dft,
                            selector_config.opencl_indexed_lmhead ? nullptr : reusable_dynamic_projector,
                            &task_result.profile,
                            selector_config.force_packed_mul_mat,
                            !selector_config.opencl_indexed_lmhead,
                            selector_config.selector_softmax_threshold_enabled ? 0 : selector_config.top_k,
                            &selector_config,
                            &runtime_bucket_state,
                            &runtime_projector_cache,
                            selector_config.opencl_indexed_lmhead ? &reusable_indexed_id_buffer : nullptr);
                    task_result.profile.task_end_us = ggml_time_us();
                } catch (...) {
                    task_error = std::current_exception();
                }

                {
                    std::lock_guard<std::mutex> lock(current_round_selection_worker_state.mutex);
                    current_round_selection_worker_state.ready_job_id = job_id;
                    current_round_selection_worker_state.result = std::move(task_result);
                    current_round_selection_worker_state.error = task_error;
                    current_round_selection_worker_state.has_ready_result = true;
                }
                current_round_selection_worker_state.cv.notify_all();
            }
        });
    }

	    auto finalize_current_round_selection_task = [&](RoundSelectionTaskResult && task_result, int64_t wait_stall_us) -> bool {
	        current_round_selection = std::move(task_result.selection);
	        current_round_selection_future_pending = false;
            current_round_selection_task_start_us = task_result.profile.task_start_us;
	        current_round_selection_task_end_us = task_result.profile.task_end_us;
	        current_round_selection_lmhead_window_recorded = false;

	        ++main_selector_rounds_completed;
	        total_main_selector_total_us += (task_result.profile.task_end_us - task_result.profile.task_start_us);
            if (task_result.profile.launch_us > 0) {
                const int64_t launch_to_dequeue_us = std::max<int64_t>(
                        0, task_result.profile.worker_dequeue_us - task_result.profile.launch_us);
                const int64_t dequeue_to_start_us = std::max<int64_t>(
                        0, task_result.profile.task_start_us - task_result.profile.worker_dequeue_us);
                const int64_t launch_to_start_us = std::max<int64_t>(
                        0, task_result.profile.task_start_us - task_result.profile.launch_us);
                const int64_t launch_to_end_us = std::max<int64_t>(
                        0, task_result.profile.task_end_us - task_result.profile.launch_us);
                total_main_selector_launch_to_dequeue_us += launch_to_dequeue_us;
                total_main_selector_dequeue_to_start_us += dequeue_to_start_us;
                total_main_selector_launch_to_start_us += launch_to_start_us;
                total_main_selector_launch_to_end_us += launch_to_end_us;
                max_main_selector_launch_to_start_us = std::max(max_main_selector_launch_to_start_us, launch_to_start_us);
                max_main_selector_launch_to_end_us = std::max(max_main_selector_launch_to_end_us, launch_to_end_us);
            }
	        total_main_selector_run_us += task_result.profile.selector.total_us;
	        total_main_selector_init_us += task_result.profile.selector.init_us;
        total_main_selector_input_write_us += task_result.profile.selector.input_write_us;
        total_main_selector_graph_execute_us += task_result.profile.selector.graph_execute_us;
        total_main_selector_output_read_us += task_result.profile.selector.output_read_us;
        total_main_selector_topk_us += task_result.profile.selector.topk_us;
        total_main_selector_shortlist_filter_us += task_result.profile.shortlist_filter_us;
        total_main_selector_projector_init_us += task_result.profile.projector_init_us;
        total_main_selector_wait_stall_us += wait_stall_us;
        total_main_selector_selected_rows += static_cast<int64_t>(current_round_selection.token_ids.size());
        total_main_selector_runtime_rows += current_round_selection.runtime_output_rows;
        if (!selector_hot_vocab_mask.empty()) {
            int64_t hot_rows = 0;
            for (const llama_token token_id : current_round_selection.token_ids) {
                if (token_id >= 0 &&
                    static_cast<size_t>(token_id) < selector_hot_vocab_mask.size() &&
                    selector_hot_vocab_mask[static_cast<size_t>(token_id)] != 0) {
                    ++hot_rows;
                }
            }
            const int64_t selected_rows = static_cast<int64_t>(current_round_selection.token_ids.size());
            const int64_t cold_rows = selected_rows - hot_rows;
            total_main_selector_hot_vocab_rows += hot_rows;
            total_main_selector_cold_vocab_rows += cold_rows;
            max_main_selector_cold_vocab_rows = std::max(max_main_selector_cold_vocab_rows, cold_rows);
        }
        max_main_selector_selected_rows = std::max<int64_t>(
                max_main_selector_selected_rows,
                static_cast<int64_t>(current_round_selection.token_ids.size()));
        max_main_selector_runtime_rows = std::max<int64_t>(
                max_main_selector_runtime_rows,
                static_cast<int64_t>(current_round_selection.runtime_output_rows));
        if (prev_main_selector_runtime_rows >= 0 &&
            prev_main_selector_runtime_rows != current_round_selection.runtime_output_rows) {
            ++total_main_selector_runtime_row_switches;
        }
        prev_main_selector_runtime_rows = current_round_selection.runtime_output_rows;

        if (current_round_selection.token_ids.empty()) {
            LOG_ERR("%s: reduced LM head path produced an empty shortlist\n", __func__);
            return false;
        }

        if (selector_config.use_reduced_lmhead &&
            current_round_selection.runtime_output_rows > 0 &&
            current_round_selection.runtime_output_source_tensor != nullptr) {
            const auto runtime_upload_start = ggml_time_us();
            if (current_round_selection.runtime_output_borrowed) {
                if (!ctx_dft->set_eagle_runtime_output_borrowed(
                            current_round_selection.runtime_output_source_tensor,
                            current_round_selection.runtime_output_rows,
                            current_round_selection.projector)) {
                    LOG_ERR("%s: failed to borrow runtime reduced LM head in draft context\n", __func__);
                    return false;
                }
                ++total_draft_runtime_output_borrowed;
            } else {
                if (!ctx_dft->set_eagle_runtime_output_copy(
                            current_round_selection.runtime_output_source_tensor,
                            current_round_selection.runtime_output_rows)) {
                    LOG_ERR("%s: failed to copy runtime reduced LM head to draft context\n", __func__);
                    return false;
                }
                ++total_draft_runtime_output_copied;
            }
            llama_set_eagle_hidden_only(ctx_dft, false);
            const auto runtime_upload_end = ggml_time_us();
            total_draft_runtime_output_upload_us += (runtime_upload_end - runtime_upload_start);
        } else if (selector_config.use_reduced_lmhead &&
                   current_round_selection.runtime_output_rows > 0 &&
                   !current_round_selection.runtime_output_weights.empty()) {
            const auto runtime_upload_start = ggml_time_us();
            if (!llama_set_eagle_runtime_output(
                        ctx_dft,
                        current_round_selection.runtime_output_weights.data(),
                        current_round_selection.runtime_output_weights.size(),
                        current_round_selection.runtime_output_rows)) {
                LOG_ERR("%s: failed to upload runtime reduced LM head to draft context\n", __func__);
                return false;
            }
            ++total_draft_runtime_output_copied;
            llama_set_eagle_hidden_only(ctx_dft, false);
            const auto runtime_upload_end = ggml_time_us();
            total_draft_runtime_output_upload_us += (runtime_upload_end - runtime_upload_start);
        } else if (selector_config.use_reduced_lmhead &&
                   selector_config.opencl_indexed_lmhead &&
                   selector_config.opencl_indexed_lmhead_in_graph &&
                   current_round_selection.projector != nullptr &&
                   current_round_selection.projector->is_opencl_indexed_mode()) {
            const auto runtime_upload_start = ggml_time_us();
            const auto & runtime_ids = current_round_selection.projector->runtime_output_indices();
            const int runtime_rows = current_round_selection.projector->runtime_output_row_capacity();
            if (runtime_rows <= 0 || runtime_ids.size() < static_cast<size_t>(runtime_rows)) {
                LOG_ERR("%s: invalid indexed runtime output ids: rows=%d ids=%zu\n",
                        __func__, runtime_rows, runtime_ids.size());
                return false;
            }
            if (!llama_set_eagle_runtime_output_ids(
                        ctx_dft,
                        runtime_ids.data(),
                        runtime_rows)) {
                LOG_ERR("%s: failed to set indexed runtime EAGLE output ids\n", __func__);
                return false;
            }
            llama_set_eagle_hidden_only(ctx_dft, false);
            const auto runtime_upload_end = ggml_time_us();
            total_draft_runtime_output_upload_us += (runtime_upload_end - runtime_upload_start);
        }

        if (selector_config.use_reduced_lmhead && current_round_selection.projector != nullptr) {
            std::string prepare_error;
            if (!current_round_selection.projector->prewarm_async(prepare_error)) {
                LOG_ERR("%s: failed to enqueue reduced projector prewarm: %s\n", __func__, prepare_error.c_str());
                return false;
            }
        }

        return true;
    };

	    auto record_current_round_selection_lmhead_window = [&](int64_t lmhead_start_us) {
	        if (current_round_selection_lmhead_window_recorded ||
	            current_round_selection_launch_us <= 0 ||
	            current_round_selection_task_end_us <= 0) {
	            return;
        }

	        const int64_t window_to_lmhead_us = std::max<int64_t>(0, lmhead_start_us - current_round_selection_launch_us);
	        const int64_t exposed_after_lmhead_us = std::max<int64_t>(0, current_round_selection_task_end_us - lmhead_start_us);
            const int64_t task_start_to_lmhead_us = current_round_selection_task_start_us > 0
                    ? std::max<int64_t>(0, lmhead_start_us - current_round_selection_task_start_us)
                    : 0;
	        total_main_selector_window_to_lmhead_us += window_to_lmhead_us;
	        total_main_selector_exposed_after_lmhead_us += exposed_after_lmhead_us;
            total_main_selector_task_start_to_lmhead_us += task_start_to_lmhead_us;
	        max_main_selector_exposed_after_lmhead_us = std::max(max_main_selector_exposed_after_lmhead_us, exposed_after_lmhead_us);
	        ++main_selector_rounds_with_lmhead_window;
        if (exposed_after_lmhead_us == 0) {
            ++main_selector_rounds_hidden_before_lmhead;
        }
        current_round_selection_lmhead_window_recorded = true;
    };

    auto try_prepare_current_round_selection = [&]() -> bool {
        if (!selector_config.use_reduced_lmhead || !current_round_selection_future_pending) {
            return true;
        }

        RoundSelectionTaskResult task_result;
        std::exception_ptr task_error;
        {
            std::lock_guard<std::mutex> lock(current_round_selection_worker_state.mutex);
            if (!current_round_selection_worker_state.has_ready_result ||
                current_round_selection_worker_state.ready_job_id != current_round_selection_job_id) {
                return true;
            }

            task_result = std::move(current_round_selection_worker_state.result);
            task_error = current_round_selection_worker_state.error;
            current_round_selection_worker_state.has_ready_result = false;
            current_round_selection_worker_state.ready_job_id = 0;
            current_round_selection_worker_state.error = nullptr;
        }

        if (task_error) {
            try {
                std::rethrow_exception(task_error);
            } catch (const std::exception & e) {
                LOG_ERR("%s: selector worker failed during early prepare: %s\n", __func__, e.what());
                return false;
            } catch (...) {
                LOG_ERR("%s: selector worker failed during early prepare with unknown exception\n", __func__);
                return false;
            }
        }

        if (!finalize_current_round_selection_task(std::move(task_result), 0)) {
            return false;
        }

        if (selector_config.debug_log_level >= 2) {
            LOG_INF("[reduced-lmhead] selector shortlist became ready before first LM head use; prewarmed early\n");
        }

        return true;
    };

    auto wait_for_current_round_selection = [&]() -> bool {
        const int64_t lmhead_start_us = ggml_time_us();

        if (current_round_selection_future_pending) {
            RoundSelectionTaskResult task_result;
            std::exception_ptr task_error;
            int64_t wait_stall_us = 0;

            {
                std::unique_lock<std::mutex> lock(current_round_selection_worker_state.mutex);
                current_round_selection_worker_state.cv.wait(lock, [&]() {
                    return current_round_selection_worker_state.has_ready_result &&
                           current_round_selection_worker_state.ready_job_id == current_round_selection_job_id;
                });

                wait_stall_us = ggml_time_us() - lmhead_start_us;
                task_result = std::move(current_round_selection_worker_state.result);
                task_error = current_round_selection_worker_state.error;
                current_round_selection_worker_state.has_ready_result = false;
                current_round_selection_worker_state.ready_job_id = 0;
                current_round_selection_worker_state.error = nullptr;
            }

            if (task_error) {
                try {
                    std::rethrow_exception(task_error);
                } catch (const std::exception & e) {
                    LOG_ERR("%s: selector worker failed: %s\n", __func__, e.what());
                    return false;
                } catch (...) {
                    LOG_ERR("%s: selector worker failed with unknown exception\n", __func__);
                    return false;
                }
            }

            if (!finalize_current_round_selection_task(std::move(task_result), wait_stall_us)) {
                return false;
            }
        }

        if (selector_config.use_reduced_lmhead && current_round_selection.token_ids.empty()) {
            LOG_ERR("%s: reduced LM head path has no prepared shortlist\n", __func__);
            return false;
        }

        record_current_round_selection_lmhead_window(lmhead_start_us);
        return true;
    };

	    auto launch_current_round_selection = [&](const std::vector<float> & hidden_input) {
	        current_round_selection = RoundSelection{};

        if (!selector_config.use_reduced_lmhead) {
            current_round_selection_future_pending = false;
            return;
        }

            const int64_t launch_submit_start_us = ggml_time_us();
	        current_round_selection_launch_us = launch_submit_start_us;
            current_round_selection_task_start_us = 0;
	        current_round_selection_task_end_us = 0;
	        current_round_selection_lmhead_window_recorded = false;

        {
            std::lock_guard<std::mutex> lock(current_round_selection_worker_state.mutex);
            if (current_round_selection_worker_state.has_pending_job ||
                current_round_selection_worker_state.has_ready_result) {
                LOG_ERR("%s: selector worker received a new launch before the previous round was consumed\n", __func__);
                current_round_selection_future_pending = false;
                return;
            }

	            current_round_selection_job_id = ++next_round_selection_job_id;
	            current_round_selection_worker_state.pending_job_id = current_round_selection_job_id;
                current_round_selection_worker_state.pending_launch_us = current_round_selection_launch_us;
	            current_round_selection_worker_state.hidden_input = hidden_input;
	            current_round_selection_worker_state.error = nullptr;
	            current_round_selection_worker_state.has_pending_job = true;
	        }

            total_main_selector_launch_submit_us += ggml_time_us() - launch_submit_start_us;
	        current_round_selection_worker_state.cv.notify_one();
        current_round_selection_future_pending = true;
        ++main_selector_rounds_launched;
    };

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
            // LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, drafts[s].tokens).c_str());
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
                const int sampled_tgt_batch_idx = drafts[s_keep].i_batch_tgt[i_dft];
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, sampled_tgt_batch_idx, true);
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

                    temp2.insert(temp2.end(),
                                 backup_data.begin() + (hidden_dim * sampled_tgt_batch_idx),
                                 backup_data.begin() + (hidden_dim * (sampled_tgt_batch_idx + 1)));
                    recompute.push_back(token_id);
                } else {
                    // greedy verification

                    // sample from the target model
                    // LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    const auto fallback_start = ggml_time_us();
                    const auto fallback_profile_start = common_sampler_profile_get();
                    token_id = common_sampler_sample(smpl, ctx_tgt, sampled_tgt_batch_idx);
                    common_sampler_profile_accumulate(
                            total_fallback_sampler_profile,
                            common_sampler_profile_diff(common_sampler_profile_get(), fallback_profile_start));

                    common_sampler_accept(smpl, token_id, true);
                    token_str = common_token_to_piece(ctx_tgt, token_id);
                    step_fallback_sampling_us += (ggml_time_us() - fallback_start);

                    temp2.insert(temp2.end(),
                                 backup_data.begin() + (hidden_dim * sampled_tgt_batch_idx),
                                 backup_data.begin() + (hidden_dim * (sampled_tgt_batch_idx + 1)));
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
        std::vector<float> temp3 = extract_selector_input_hidden(backup_data, hidden_dim);
        if (temp3.size() != static_cast<size_t>(hidden_dim)) {
            LOG_ERR("%s: failed to extract selector input hidden from backup_data (size=%zu hidden_dim=%d)\n",
                    __func__, backup_data.size(), hidden_dim);
            return 1;
        }
        if (!selector_config.selector_launch_after_recompute) {
            launch_current_round_selection(temp3);
        }
        int recompute_point = n_past_dft - i_dft;

        topk_indices = { 0, };

        /////////////////////////////////////////Drafting Start///////////////////////////////////////

        const auto drafting_start = ggml_time_us(); //tree decoding 시작 시간 기록 -ym-
        // LOG_DBG("Current n_accept: %d, n_drafted: %d, n_predict: %d\n", n_accept, n_drafted, n_predict);

        //////////////////////////////////////////Recompute Logic Start////////////////////////////////////////
        if (selector_config.use_reduced_lmhead) {
            // Recompute should stay on the hidden-only EAGLE graph, matching the static trimmed path.
            // The runtime reduced LM head is only needed for the later tree expansion rounds.
            llama_set_eagle_hidden_only(ctx_dft, true);
        }
        const auto step_recompute_start = ggml_time_us();
        {
            const auto recompute_state_reset_start = ggml_time_us();
            // LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());
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
        if (selector_config.selector_launch_after_recompute) {
            launch_current_round_selection(temp3);
        }
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

        if (!try_prepare_current_round_selection()) {
            return 1;
        }

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
            std::vector<std::vector<float>> reduced_seq_raw_logits;
            const auto step_loop_prep_end = ggml_time_us();
            total_expansion_loop_prep_us += (step_loop_prep_end - step_loop_prep_start);

            if (selector_config.use_reduced_lmhead) {
                if (!try_prepare_current_round_selection()) {
                    return 1;
                }
                if (!wait_for_current_round_selection()) {
                    return 1;
                }

                const bool capture_root_raw =
                        (i == 0 && selector_config.debug_log_level >= 2 && selector_config.dump_reduced_logits);
                const bool capture_compare_raw =
                        (i == 0 && debug_trimmed_ref.enabled && selector_config.debug_log_level >= 2);
                const bool capture_raw_logits = capture_root_raw || capture_compare_raw;
                if (capture_root_raw) {
                    reduced_seq_raw_logits.resize(n_seq_dft);
                }

                const auto selector_prep_start = ggml_time_us();
                std::vector<int> active_seq_ids;
                std::vector<int> active_batch_indices;
                std::vector<struct common_sampler *> active_samplers;
                std::vector<struct common_sampler *> compare_samplers;
                std::vector<float> hidden_batch;
                active_seq_ids.reserve(n_seq_dft);
                active_batch_indices.reserve(n_seq_dft);
                active_samplers.reserve(n_seq_dft);
                compare_samplers.reserve(n_seq_dft);

                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].drafting || drafts[s].skip) {
                        continue;
                    }

                    active_seq_ids.push_back(s);
                    active_batch_indices.push_back(drafts[s].i_batch_dft);
                    active_samplers.push_back(drafts[s].smpl);
                    compare_samplers.push_back(capture_compare_raw ? common_sampler_clone(drafts[s].smpl) : nullptr);
                    if (i == 0 || selector_config.opencl_indexed_lmhead) {
                        const size_t hidden_offset = static_cast<size_t>(hidden_dim) * drafts[s].i_batch_dft;
                        if (hidden_offset + hidden_dim > cb_data.data.size()) {
                            LOG_ERR("%s: missing draft hidden for seq=%d i_batch_dft=%d (cb_data=%zu hidden_dim=%d)\n",
                                    __func__, s, drafts[s].i_batch_dft, cb_data.data.size(), hidden_dim);
                            return 1;
                        }
                        const float * branch_hidden = cb_data.data.data() + hidden_offset;
                        hidden_batch.insert(hidden_batch.end(), branch_hidden, branch_hidden + hidden_dim);
                    }
                }
                const auto selector_prep_end = ggml_time_us();
                total_expansion_selector_prep_us += (selector_prep_end - selector_prep_start);

                if (!active_seq_ids.empty()) {
                    const auto t_samp_start = ggml_time_us();
                    std::vector<std::vector<float>> active_raw_logits;
                    ReducedDraftSamplingProfile reduced_sampling_profile = {};
                    std::string reduced_error;
                    const bool indexed_graph_logits_active =
                            selector_config.opencl_indexed_lmhead &&
                            selector_config.opencl_indexed_lmhead_in_graph &&
                            current_round_selection.projector != nullptr &&
                            current_round_selection.projector->is_opencl_indexed_mode();
                    const bool use_projector_logits =
                            i == 0 ||
                            (selector_config.opencl_indexed_lmhead &&
                             !indexed_graph_logits_active);
                    const bool reduced_ok = use_projector_logits
                            ? compute_candidates_from_projector_batch(
                                    hidden_batch.data(),
                                    static_cast<int>(active_seq_ids.size()),
                                    current_round_selection,
                                    active_samplers,
                                    capture_raw_logits ? &active_raw_logits : nullptr,
                                    &reduced_sampling_profile,
                                    reduced_error)
                            : compute_candidates_from_fused_ctx_logits_batch(
                                    ctx_dft,
                                    current_round_selection,
                                    active_batch_indices,
                                    active_samplers,
                                    capture_raw_logits ? &active_raw_logits : nullptr,
                                    &reduced_sampling_profile,
                                    reduced_error);
                    if (!reduced_ok) {
                        for (auto * sampler_clone : compare_samplers) {
                            if (sampler_clone != nullptr) {
                                common_sampler_free(sampler_clone);
                            }
                        }
                        LOG_ERR("%s: batched reduced projection failed at depth=%d: %s\n",
                                __func__, i, reduced_error.c_str());
                        return 1;
                    }
                    const auto t_samp_end = ggml_time_us();
                    total_expansion_sampling_us += (t_samp_end - t_samp_start);
                    total_reduced_draft_logits_compute_us += reduced_sampling_profile.logits_compute_us;
                    total_reduced_fused_logits_fetch_us += reduced_sampling_profile.logits_fetch_us;
                    if (use_projector_logits) {
                        total_reduced_draft_sampler_apply_us += reduced_sampling_profile.sampler_apply_us;
                    } else {
                        total_reduced_fused_sampler_apply_us += reduced_sampling_profile.sampler_apply_us;
                    }

                    const auto selector_postprocess_start = ggml_time_us();
                    if (capture_compare_raw) {
                        debug_compare_reduced_batch_with_trimmed(
                                current_round_selection,
                                active_seq_ids,
                                active_samplers,
                                compare_samplers,
                                hidden_batch,
                                active_raw_logits,
                                debug_trimmed_ref,
                                i,
                                selector_config.debug_log_level >= 2);
                    }

                    for (auto * sampler_clone : compare_samplers) {
                        if (sampler_clone != nullptr) {
                            common_sampler_free(sampler_clone);
                        }
                    }

                    if (capture_root_raw) {
                        for (size_t active_idx = 0; active_idx < active_seq_ids.size(); ++active_idx) {
                            const int seq_id = active_seq_ids[active_idx];
                            reduced_seq_raw_logits[seq_id] = std::move(active_raw_logits[active_idx]);
                        }
                    }

                    if (capture_root_raw && !reduced_seq_raw_logits.empty() && !reduced_seq_raw_logits[0].empty()) {
                        const float * root_hidden = hidden_batch.empty() ? nullptr : hidden_batch.data();
                        if (debug_trimmed_ref.enabled) {
                            debug_compare_reduced_logits_with_trimmed(
                                    current_round_selection,
                                    reduced_seq_raw_logits[0],
                                    root_hidden,
                                    debug_trimmed_ref,
                                    ctx_dft,
                                    "root-reduced-compare",
                                    selector_config.debug_log_level >= 2);
                        }
                        const auto * root_candidates = common_sampler_get_candidates(drafts[0].smpl, true);
                        if (root_candidates != nullptr && root_candidates->data != nullptr && root_candidates->size > 0) {
                            std::vector<llama_token_data> root_dist(
                                    root_candidates->data,
                                    root_candidates->data + root_candidates->size);
                            print_candidate_distribution("root-reduced", ctx_dft, root_dist);
                        }
                    }
                    const auto selector_postprocess_end = ggml_time_us();
                    total_expansion_selector_postprocess_us += (selector_postprocess_end - selector_postprocess_start);
                }
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                // LOG_DBG("drafting sequence %d at pos %d\n", s, i);

                ////////////////////////////////////////Sampling Start///////////////////////////////////////

                std::vector<llama_token_data> cur_candidates;
                if (selector_config.use_reduced_lmhead) {
                    const auto candidate_fetch_start = ggml_time_us();
                    const auto * draft_candidates = common_sampler_get_candidates(drafts[s].smpl, true);
                    if (draft_candidates != nullptr && draft_candidates->data != nullptr && draft_candidates->size > 0) {
                        cur_candidates.assign(draft_candidates->data, draft_candidates->data + draft_candidates->size);
                    }
                    const auto candidate_fetch_end = ggml_time_us();
                    total_expansion_candidate_fetch_us += (candidate_fetch_end - candidate_fetch_start);
                } else {
                    const auto t_samp_start = ggml_time_us();
                    common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                    const auto * draft_candidates = common_sampler_get_candidates(drafts[s].smpl, true);
                    cur_candidates.assign(draft_candidates->data, draft_candidates->data + draft_candidates->size);

                    // EAGLE vocab trimming: remap candidate IDs from trimmed logits index to original token IDs
                    if (has_vocab_trim) {
                        for (auto & candidate : cur_candidates) {
                            const int idx = candidate.id;
                            if (idx >= 0 && idx < (int)dft_vocab_map.size()) {
                                candidate.id = dft_vocab_map[idx];
                            }
                        }
                    }
                    const auto t_samp_end = ggml_time_us();
                    total_expansion_sampling_us += (t_samp_end - t_samp_start);
                }

                if (cur_candidates.empty()) {
                    LOG_ERR("%s: no draft candidates available for seq=%d depth=%d (reduced=%s)\n",
                            __func__, s, i, selector_config.use_reduced_lmhead ? "true" : "false");
                    return 1;
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
                float prob = cur_candidates[0].p;
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
                const int split_limit = std::min<int>(expand_k, cur_candidates.size());
                for (int f = 1; f < split_limit; ++f) {
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
                        float prob = cur_candidates[f].p;
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
                    const llama_token id = cur_candidates[is].id;
                    ids.push_back(id);
                    temp_probs.push_back(cur_candidates[is].p);
                    datas.push_back(cur_candidates);

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
                const auto & cur_dist = datas[is];

                common_sampler_accept(drafts[s].smpl, id, true);
                drafts[s].tokens.push_back(id);
                drafts[s].dists.push_back(cur_dist);

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

                const float * parent_hidden = nullptr;
                if (selector_config.use_reduced_lmhead && i == 0) {
                    parent_hidden = temp3.data();
                } else {
                    const size_t hidden_offset = static_cast<size_t>(hidden_dim) * temp_i_batch_dft[s];
                    if (hidden_offset + hidden_dim > cb_data.data.size()) {
                        LOG_ERR("%s: missing parent hidden for seq=%d temp_i_batch_dft=%d (cb_data=%zu hidden_dim=%d)\n",
                                __func__, s, temp_i_batch_dft[s], cb_data.data.size(), hidden_dim);
                        return 1;
                    }
                    parent_hidden = cb_data.data.data() + hidden_offset;
                }

                if (topk_indices.size() == 1) {
                    common_batch_add(batch_dft, id, n_past_cur, {s}, true);
                    temp.insert(temp.end(), parent_hidden, parent_hidden + hidden_dim);
                }
                else {
                    auto it_last = std::find(topk_indices.begin(), topk_indices.end(), s);
                    if (it_last != topk_indices.end()) {
                        common_batch_add(batch_dft, id, n_past_cur, {s}, true);
                        temp.insert(temp.end(), parent_hidden, parent_hidden + hidden_dim);
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
            total_draft_forward_us        += (dft_model_decode_end        - dft_model_decode_start);
            total_draft_forward_submit_us += (dft_model_decode_submit_end - dft_model_decode_start);
            total_draft_forward_sync_us   += (dft_model_decode_end        - dft_model_decode_submit_end);
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
                if (ctx_tgt->final_hiddens.size() % hidden_dim != 0) {
                    LOG_ERR("%s: invalid verification final_hiddens size %zu for hidden_dim=%d\n",
                            __func__, ctx_tgt->final_hiddens.size(), hidden_dim);
                    break;
                }
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
        const int64_t reduced_target_fallback_other_us = std::max<int64_t>(
                0,
                total_fallback_sampling_us
                    - total_reduced_target_shortlist_build_us
                    - total_reduced_target_sampler_apply_us
                    - total_reduced_target_accept_piece_us);
        const int64_t fallback_sampler_other_us = std::max<int64_t>(
                0,
                total_fallback_sampling_us - total_fallback_sampler_profile.sample_total_us);
        const int64_t fallback_sample_calls = total_fallback_sampler_profile.sample_calls;
        const double main_selector_overlap_before_lmhead_pct =
                (total_main_selector_total_us > 0 && main_selector_rounds_with_lmhead_window > 0)
                ? 100.0 * (static_cast<double>(total_main_selector_total_us - total_main_selector_exposed_after_lmhead_us) /
                           static_cast<double>(total_main_selector_total_us))
                : 0.0;

        const auto avg_ms_per = [](int64_t total_us, int64_t count) -> double {
            return count > 0 ? static_cast<double>(total_us) / 1000.0 / static_cast<double>(count) : 0.0;
        };
        const auto avg_per = [](int64_t total, int64_t count) -> double {
            return count > 0 ? static_cast<double>(total) / static_cast<double>(count) : 0.0;
        };
        const int64_t total_expansion_split_us =
                total_split_kv_copy_us + total_split_history_update_us + total_split_draft_state_alloc_us;
        const int64_t total_reduced_lmhead_apply_us =
                total_expansion_selector_prep_us +
                total_expansion_sampling_us +
                total_expansion_selector_postprocess_us +
                total_expansion_candidate_fetch_us;
        const int64_t total_expansion_core_us =
                total_expansion_loop_prep_us +
                total_expansion_candidate_bookkeeping_us +
                total_expansion_split_us +
                total_expansion_temp_probs_us +
                total_expansion_topk_us +
                total_expansion_target_batch_us +
                total_expansion_post_batch_us;
        const int64_t total_expansion_us =
                total_reduced_lmhead_apply_us +
                total_expansion_core_us;
        const int64_t total_main_selector_qnn_us =
                total_main_selector_input_write_us +
                total_main_selector_graph_execute_us +
                total_main_selector_output_read_us;
        const int64_t total_main_selector_post_qnn_us =
                total_main_selector_topk_us +
                total_main_selector_shortlist_filter_us +
                total_main_selector_projector_init_us;
        const int64_t total_main_selector_run_other_us = std::max<int64_t>(
                0,
                total_main_selector_run_us -
                    total_main_selector_init_us -
                    total_main_selector_qnn_us -
                    total_main_selector_topk_us);
        const int64_t total_main_selector_round_other_us = std::max<int64_t>(
                0,
                total_main_selector_total_us -
                    total_main_selector_run_us -
                    total_main_selector_shortlist_filter_us -
                    total_main_selector_projector_init_us);
        const int64_t total_draft_accounted_us =
                total_draft_recompute_us +
                total_draft_setup_us +
                total_main_selector_wait_stall_us +
                total_expansion_us +
                total_draft_forward_us +
                total_tree_pruning_us;
        const int64_t total_draft_unaccounted_us = std::max<int64_t>(0, total_draft_wall_us - total_draft_accounted_us);

        LOG_INF("\n");
        LOG_INF("======= Latency Breakdown (Avg / Step, Additive) =========\n");
        LOG_INF("Prefill (one-time)              : %8.3f ms\n", prefill_ms);
        LOG_INF("Decode steps                    : %8d\n", num_steps);
        LOG_INF("Selector rounds                 : %8d completed\n", main_selector_rounds_completed);
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
        if (selector_config.use_reduced_lmhead) {
            LOG_INF("    * Runtime Output Upload     : %8.3f ms\n", avg_ms_per(total_draft_runtime_output_upload_us, num_steps));
            LOG_INF("      - Borrowed zero-copy      : %8lld / %d steps\n",
                    (long long) total_draft_runtime_output_borrowed,
                    num_steps);
            LOG_INF("      - Explicit copy/upload    : %8lld / %d steps\n",
                    (long long) total_draft_runtime_output_copied,
                    num_steps);
        }
        LOG_INF("  - Selector Wait Stall         : %8.3f ms\n", avg_ms_per(total_main_selector_wait_stall_us, num_steps));
        if (selector_config.use_reduced_lmhead) {
            LOG_INF("  - Reduced LMHead Apply        : %8.3f ms\n", avg_ms_per(total_reduced_lmhead_apply_us, num_steps));
            LOG_INF("    * Hidden Gather/Prep        : %8.3f ms\n", avg_ms_per(total_expansion_selector_prep_us, num_steps));
            LOG_INF("    * Reduced Logits Total      : %8.3f ms\n", avg_ms_per(total_expansion_sampling_us, num_steps));
            if (total_reduced_draft_logits_compute_us > 0 ||
                total_reduced_fused_logits_fetch_us > 0 ||
                total_reduced_draft_sampler_apply_us > 0 ||
                total_reduced_fused_sampler_apply_us > 0) {
                LOG_INF("      - Projector Compute       : %8.3f ms\n", avg_ms_per(total_reduced_draft_logits_compute_us, num_steps));
                LOG_INF("      - Fused Logits Fetch      : %8.3f ms\n", avg_ms_per(total_reduced_fused_logits_fetch_us, num_steps));
                LOG_INF("      - Projector Sampler Apply : %8.3f ms\n", avg_ms_per(total_reduced_draft_sampler_apply_us, num_steps));
                LOG_INF("      - Fused Sampler Apply     : %8.3f ms\n", avg_ms_per(total_reduced_fused_sampler_apply_us, num_steps));
            }
            LOG_INF("    * Candidate Fetch           : %8.3f ms\n", avg_ms_per(total_expansion_candidate_fetch_us, num_steps));
            LOG_INF("    * Selector Postprocess      : %8.3f ms\n", avg_ms_per(total_expansion_selector_postprocess_us, num_steps));
            LOG_INF("  - Tree Expansion Core         : %8.3f ms\n", avg_ms_per(total_expansion_core_us, num_steps));
            LOG_INF("    * Loop Prep                 : %8.3f ms\n", avg_ms_per(total_expansion_loop_prep_us, num_steps));
            LOG_INF("    * Candidate Bookkeeping     : %8.3f ms\n", avg_ms_per(total_expansion_candidate_bookkeeping_us, num_steps));
            LOG_INF("    * Split Sequence            : %8.3f ms\n", avg_ms_per(total_expansion_split_us, num_steps));
            LOG_INF("      - KV Cache Copy           : %8.3f ms\n", avg_ms_per(total_split_kv_copy_us, num_steps));
            LOG_INF("      - Seq History Update      : %8.3f ms\n", avg_ms_per(total_split_history_update_us, num_steps));
            LOG_INF("      - Draft State Alloc       : %8.3f ms\n", avg_ms_per(total_split_draft_state_alloc_us, num_steps));
            LOG_INF("    * Temp Probs Array Prep     : %8.3f ms\n", avg_ms_per(total_expansion_temp_probs_us, num_steps));
            LOG_INF("    * TopK Sorting              : %8.3f ms\n", avg_ms_per(total_expansion_topk_us, num_steps));
            LOG_INF("    * Target Batch Append       : %8.3f ms\n", avg_ms_per(total_expansion_target_batch_us, num_steps));
            LOG_INF("    * Post Batch Bookkeeping    : %8.3f ms\n", avg_ms_per(total_expansion_post_batch_us, num_steps));
        } else {
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
        }
        LOG_INF("  - Draft Tree Forward          : %8.3f ms\n", avg_ms_per(total_draft_forward_us, num_steps));
        LOG_INF("    * Pre-sync submit           : %8.3f ms\n", avg_ms_per(total_draft_forward_submit_us, num_steps));
        LOG_INF("    * Post-submit sync wait     : %8.3f ms\n", avg_ms_per(total_draft_forward_sync_us, num_steps));
        LOG_INF("    * Tree decode calls         : %8.3f / step (%d total)\n",
                num_steps > 0 ? (double) total_draft_forward_calls / (double) num_steps : 0.0,
                total_draft_forward_calls);
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
        LOG_INF("[1a] Async Selector Round (reference, overlaps with drafting)\n");
        if (main_selector_rounds_completed > 0) {
            LOG_INF("  - Main Selector               : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_total_us, main_selector_rounds_completed));
            LOG_INF("    * Selector Run Total        : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_run_us, main_selector_rounds_completed));
            if (selector_predecode_init_us > 0) {
                LOG_INF("      - Selector Init Preload   : %8.3f ms one-time (outside decode loop)\n",
                        selector_predecode_init_us / 1000.0);
            }
            LOG_INF("      - Selector Init Lazy      : %8.3f ms / selector round (%8.3f ms total)\n",
                    avg_ms_per(total_main_selector_init_us, main_selector_rounds_completed),
                    total_main_selector_init_us / 1000.0);
            LOG_INF("    * Selector QNN              : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_qnn_us, main_selector_rounds_completed));
            LOG_INF("      - QNN Input Write         : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_input_write_us, main_selector_rounds_completed));
            LOG_INF("      - QNN Graph Execute       : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_graph_execute_us, main_selector_rounds_completed));
            LOG_INF("      - QNN Output Read         : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_output_read_us, main_selector_rounds_completed));
            LOG_INF("    * Selector Post-QNN         : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_post_qnn_us, main_selector_rounds_completed));
            LOG_INF("      - Selector Select (%s): %8.3f ms / selector round\n",
                    selector_config.selector_softmax_threshold_enabled ? "softmax-p" : "top-k",
                    avg_ms_per(total_main_selector_topk_us, main_selector_rounds_completed));
            LOG_INF("      - Shortlist Filter/Map    : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_shortlist_filter_us, main_selector_rounds_completed));
            LOG_INF("      - Reduced Projector Init  : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_projector_init_us, main_selector_rounds_completed));
            LOG_INF("      - Selector Run Other      : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_run_other_us, main_selector_rounds_completed));
            LOG_INF("      - Round Build Other       : %8.3f ms / selector round\n",
                    avg_ms_per(total_main_selector_round_other_us, main_selector_rounds_completed));
            LOG_INF("    * Selected Rows             : %8.3f / round (max %lld)\n",
                    avg_per(total_main_selector_selected_rows, main_selector_rounds_completed),
                    (long long) max_main_selector_selected_rows);
            if (!selector_hot_vocab_mask.empty()) {
                LOG_INF("    * Hot Vocab Coverage        : hot %8.3f / round (%6.2f%%), cold %8.3f / round (max %lld)\n",
                        avg_per(total_main_selector_hot_vocab_rows, main_selector_rounds_completed),
                        total_main_selector_selected_rows > 0
                                ? 100.0 * static_cast<double>(total_main_selector_hot_vocab_rows) /
                                          static_cast<double>(total_main_selector_selected_rows)
                                : 0.0,
                        avg_per(total_main_selector_cold_vocab_rows, main_selector_rounds_completed),
                        (long long) max_main_selector_cold_vocab_rows);
            }
            LOG_INF("    * Runtime Output Rows       : %8.3f / round (max %lld)\n",
                    avg_per(total_main_selector_runtime_rows, main_selector_rounds_completed),
                    (long long) max_main_selector_runtime_rows);
            if (selector_config.runtime_bucket_enabled) {
                LOG_INF("    * Runtime Row Buckets       : enabled (shrink_ratio=%.3f patience=%d)\n",
                        static_cast<double>(selector_config.runtime_bucket_shrink_ratio),
                        selector_config.runtime_bucket_shrink_patience);
                LOG_INF("      - Bucket switches         : %8lld / %d rounds\n",
                        (long long) total_main_selector_runtime_row_switches,
                        main_selector_rounds_completed);
                LOG_INF("      - Projector cache entries : %8zu (limit %d)\n",
                        runtime_projector_cache.size(),
                        selector_config.projector_cache_limit);
            }
            LOG_INF("    * Selector Launch Timing    : %s\n",
                    selector_config.selector_launch_after_recompute ? "after-recompute" : "before-recompute");
            if (main_selector_rounds_launched > 0) {
                LOG_INF("    * Launch Submit Cost        : %8.3f ms / launch\n",
                        avg_ms_per(total_main_selector_launch_submit_us, main_selector_rounds_launched));
            }
            LOG_INF("    * Launch -> Worker Start    : %8.3f ms / round (max %8.3f)\n",
                    avg_ms_per(total_main_selector_launch_to_start_us, main_selector_rounds_completed),
                    max_main_selector_launch_to_start_us / 1000.0);
            LOG_INF("      - Launch -> Dequeue       : %8.3f ms / round\n",
                    avg_ms_per(total_main_selector_launch_to_dequeue_us, main_selector_rounds_completed));
            LOG_INF("      - Dequeue -> Task Start   : %8.3f ms / round\n",
                    avg_ms_per(total_main_selector_dequeue_to_start_us, main_selector_rounds_completed));
            LOG_INF("    * Launch -> Task End        : %8.3f ms / round (max %8.3f)\n",
                    avg_ms_per(total_main_selector_launch_to_end_us, main_selector_rounds_completed),
                    max_main_selector_launch_to_end_us / 1000.0);
            if (main_selector_rounds_with_lmhead_window > 0) {
                LOG_INF("    * Window -> LMHead          : %8.3f ms / window\n",
                        avg_ms_per(total_main_selector_window_to_lmhead_us, main_selector_rounds_with_lmhead_window));
                LOG_INF("    * TaskStart -> LMHead       : %8.3f ms / window\n",
                        avg_ms_per(total_main_selector_task_start_to_lmhead_us, main_selector_rounds_with_lmhead_window));
                LOG_INF("    * Exposed After LMHead      : %8.3f ms / window\n",
                        avg_ms_per(total_main_selector_exposed_after_lmhead_us, main_selector_rounds_with_lmhead_window));
                LOG_INF("    * Hidden Before LMHead      : %8d / %d windows\n",
                        main_selector_rounds_hidden_before_lmhead,
                        main_selector_rounds_with_lmhead_window);
                LOG_INF("    * Overlap Before LMHead     : %8.2f %%\n", main_selector_overlap_before_lmhead_pct);
            }
        }
        LOG_INF("[1b] Draft Depth Latency (avg / visit)\n");
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
        if (selector_config.use_reduced_lmhead && total_reduced_target_apply_calls > 0) {
            LOG_INF("    * Reduced shortlist build   : %8.3f ms / step | %8.3f ms / call\n",
                    avg_ms_per(total_reduced_target_shortlist_build_us, num_steps),
                    avg_ms_per(total_reduced_target_shortlist_build_us, total_reduced_target_apply_calls));
            LOG_INF("    * Reduced sampler apply     : %8.3f ms / step | %8.3f ms / call\n",
                    avg_ms_per(total_reduced_target_sampler_apply_us, num_steps),
                    avg_ms_per(total_reduced_target_sampler_apply_us, total_reduced_target_apply_calls));
            LOG_INF("    * Reduced accept/piece      : %8.3f ms / step | %8.3f ms / call\n",
                    avg_ms_per(total_reduced_target_accept_piece_us, num_steps),
                    avg_ms_per(total_reduced_target_accept_piece_us, total_reduced_target_apply_calls));
            LOG_INF("    * Reduced other             : %8.3f ms / step\n",
                    avg_ms_per(reduced_target_fallback_other_us, num_steps));
            LOG_INF("    * Reduced apply calls       : %8lld\n", (long long) total_reduced_target_apply_calls);
        }
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
        fprintf(stderr, "  Draft path mode       : %s\n",
                selector_config.use_reduced_lmhead ? "reduced-lmhead" : "baseline-trimmed/full");
        fprintf(stderr, "  Draft length (depth)  : %.3f\n", draft_len);
        fprintf(stderr, "  Draft tokens/step     : %.3f\n", draft_tokens_avg);
        fprintf(stderr, "  Avg accept length     : %.3f\n", accept_len);
        fprintf(stderr, "  Accept ratio          : %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
        if (main_selector_rounds_completed > 0) {
            fprintf(stderr, "  Avg selected rows     : %.3f / round (max %lld)\n",
                    static_cast<double>(total_main_selector_selected_rows) / static_cast<double>(main_selector_rounds_completed),
                    (long long) max_main_selector_selected_rows);
            if (!selector_hot_vocab_mask.empty()) {
                fprintf(stderr, "  Hot vocab coverage    : hot %.3f / round (%.2f%%), cold %.3f / round (max %lld)\n",
                        static_cast<double>(total_main_selector_hot_vocab_rows) / static_cast<double>(main_selector_rounds_completed),
                        total_main_selector_selected_rows > 0
                                ? 100.0 * static_cast<double>(total_main_selector_hot_vocab_rows) /
                                          static_cast<double>(total_main_selector_selected_rows)
                                : 0.0,
                        static_cast<double>(total_main_selector_cold_vocab_rows) / static_cast<double>(main_selector_rounds_completed),
                        (long long) max_main_selector_cold_vocab_rows);
            }
            fprintf(stderr, "  Avg runtime rows      : %.3f / round (max %lld)\n",
                    static_cast<double>(total_main_selector_runtime_rows) / static_cast<double>(main_selector_rounds_completed),
                    (long long) max_main_selector_runtime_rows);
        }
        fprintf(stderr, "------------------------------------------------------------\n");
        fprintf(stderr, "  Avg draft phase       : %9.3f ms\n", avg_draft_lat);
        fprintf(stderr, "  Avg verification      : %9.3f ms\n", avg_verify_lat);
        fprintf(stderr, "  Target batch toks     : %9.3f / step (max %d)\n",
                avg_target_batch_tokens,
                max_target_batch_tokens);
        fprintf(stderr, "  Target seq refs       : %9.3f / step (%.3f / tok, max %d)\n",
                avg_target_batch_seq_refs,
                avg_target_batch_seq_refs_per_token,
                max_target_batch_seq_refs_per_token);
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

    if (main_selector_rounds_completed > 0) {
        LOG_INF("\n");
        LOG_INF("Selector shortlist stats:\n");
        LOG_INF("  Avg selected rows: %.3f / round (max %lld)\n",
                static_cast<double>(total_main_selector_selected_rows) / static_cast<double>(main_selector_rounds_completed),
                (long long) max_main_selector_selected_rows);
        if (!selector_hot_vocab_mask.empty()) {
            LOG_INF("  Hot vocab coverage: hot %.3f / round (%.2f%%), cold %.3f / round (max %lld)\n",
                    static_cast<double>(total_main_selector_hot_vocab_rows) / static_cast<double>(main_selector_rounds_completed),
                    total_main_selector_selected_rows > 0
                            ? 100.0 * static_cast<double>(total_main_selector_hot_vocab_rows) /
                                      static_cast<double>(total_main_selector_selected_rows)
                            : 0.0,
                    static_cast<double>(total_main_selector_cold_vocab_rows) / static_cast<double>(main_selector_rounds_completed),
                    (long long) max_main_selector_cold_vocab_rows);
        }
        LOG_INF("  Avg runtime rows : %.3f / round (max %lld)\n",
                static_cast<double>(total_main_selector_runtime_rows) / static_cast<double>(main_selector_rounds_completed),
                (long long) max_main_selector_runtime_rows);
    }

    if (!decoding_latencies.empty() && !verification_latencies.empty()) {
        const double avg_decoding_latency = std::accumulate(decoding_latencies.begin(), decoding_latencies.end(), 0.0) / decoding_latencies.size();
        const double avg_verification_latency = std::accumulate(verification_latencies.begin(), verification_latencies.end(), 0.0) / verification_latencies.size();
        LOG_INF("\navg drafting latency: %.3f ms\n", avg_decoding_latency);
        LOG_INF("avg verification latency: %.3f ms\n", avg_verification_latency);
        LOG_INF("avg T_d: %.3f ms\n", !T_d.empty() ? (std::accumulate(T_d.begin(), T_d.end(), 0.0) / T_d.size()) : 0.0);
        LOG_INF("Verification/Draft Phase Count: %zu\n", verification_latencies.size());
    }

    // Accepted Token Counts Matrix 출력 (디버깅용)
    // 너무 많은 row를 출력해서 일반 실행 로그를 묻기 때문에 필요할 때만 임시로 되살립니다.
    // LOG_INF("\nAccepted Token Counts Matrix:\n");
    // for (int i = 0; i < rows; i++) {
    //     LOG_INF("[");
    //     for (int j = 0; j < cols; j++) {
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
