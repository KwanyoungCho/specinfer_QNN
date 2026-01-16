#pragma once

#include "qnn_loader.h"
#include "qnn_qnnjson.h"
#include "qnn_tensor_util.h"
#include "io_alloc.h"
#include "llm_kv_cache_manager.h"
#include "llm_kv_cache_mapper.h"
#include "llm_output_processor.h"
#include "model_params.h"
#include "../llama-context.h"

#include <string>
#include <vector>
#include <memory>

namespace llama_qnn {

/**
 * @brief Configuration for LLM Decode Runner
 */
struct LLMDecodeConfig {
  std::string ctx_dir;          // QNN context directory
  std::string backend_so;       // QNN backend library path
  std::string system_so;        // QNN system library path (optional)
  std::string tokenizer_path;   // Tokenizer model path
  std::string params_path;      // params.json path (optional, for dynamic config)
  int max_gen_tokens = 100;     // Maximum tokens to generate
  int log_level = 0;            // 0=quiet, 1=info, 2=debug
  bool use_multi_context = false; // Enable multi-context (sharding) mode
  int num_shards = 0;           // Number of context shards (0=auto-detect, default)
};

/**
 * @brief High-level API for LLM Prefill + Decode execution
 * 
 * Manages:
 * - QNN context loading and graph execution
 * - KV cache allocation and mapping
 * - Prefill → Decode transition (rearrange_cache)
 * - Token generation loop
 */
class LLMDecodeRunner {
 public:
  explicit LLMDecodeRunner(const LLMDecodeConfig& config);
  ~LLMDecodeRunner();
  
  /**
   * @brief Initialize QNN backend and load graphs
   * @return true on success
   */
  bool initialize();
  
  /**
   * @brief Get last error message
   */
  const std::string& get_error() const { return error_msg_; }
  
  // ========== KV Cache Metadata API (llama_memory_seq_* compatible) ==========
  
  /**
   * @brief Remove seq_id from cells in range [p0, p1)
   */
  void kv_seq_rm(int32_t seq_id, int32_t p0, int32_t p1);
  
  /**
   * @brief Keep only cells that have seq_id
   */
  void kv_seq_keep(int32_t seq_id);
  
  /**
   * @brief Copy src_seq to dst_seq for cells in range [p0, p1)
   */
  void kv_seq_cp(int32_t src_seq, int32_t dst_seq, int32_t p0, int32_t p1);
  
  /**
   * @brief Get KV manager (for advanced operations)
   */
  LLMKVCacheManager* get_kv_manager() { return kv_manager_.get(); }
  
 private:
  // Configuration
  LLMDecodeConfig config_;
  std::string error_msg_;
  
  // QNN components
  std::unique_ptr<QnnLoader> loader_;
  
  // Single-context mode
  std::map<std::string, QnnJsonGraphDesc> graphs_;
  QnnJsonGraphDesc* prefill_graph_;
  QnnJsonGraphDesc* kv_graph_;
  
  // Multi-context mode (sharding)
  struct ShardInfo {
    std::map<std::string, QnnJsonGraphDesc> graphs;
    QnnJsonGraphDesc* prefill_graph;
    QnnJsonGraphDesc* kv_graph;
    
    // I/O allocators for this shard
    std::unique_ptr<QNNIOAllocator> prefill_alloc;
    std::unique_ptr<QNNIOAllocator> kv_alloc;
    
    // Tensor holders for zero-copy binding
    std::vector<std::unique_ptr<QnnTensorHolder>> prefill_input_holders;
    std::vector<std::unique_ptr<QnnTensorHolder>> prefill_output_holders;
    std::vector<std::unique_ptr<QnnTensorHolder>> kv_input_holders;
    std::vector<std::unique_ptr<QnnTensorHolder>> kv_output_holders;
  };
  std::vector<ShardInfo> shards_;
  
  // Shared buffers across shards
  std::map<std::string, void*> shared_buffer_views_; // hidden_state, rope_cos, rope_sin, attention_mask
  
  // Model metadata
  ModelParams model_params_;    // Parsed from params.json
  int context_len_;
  int num_layers_;
  int num_heads_;
  int head_dim_;
  int prefill_ar_len_;
  int kv_ar_len_;
  int prefill_cache_len_;
  int kv_cache_len_;
  int layers_per_shard_;        // For multi-context: n_layers / num_shards
  int32_t n_past_;               // Internal position tracking for qnn_decode
  bool prefill_done_;             // Flag to track prefill→decode transition
  
  // KV cache
  std::unique_ptr<LLMKVCacheManager> kv_manager_;
  std::vector<KVCacheTensorInfo> prefill_kv_mapping_;
  std::vector<KVCacheTensorInfo> kv_kv_mapping_;
  std::map<std::string, void*> prefill_kv_override_;
  std::map<std::string, void*> kv_kv_override_;
  
  // I/O allocators (single-context only)
  std::unique_ptr<QNNIOAllocator> prefill_alloc_;
  std::unique_ptr<QNNIOAllocator> kv_alloc_;
  
  // Pre-built QNN tensors (single-context only, reused across executions)
  std::vector<std::unique_ptr<QnnTensorHolder>> prefill_input_holders_;
  std::vector<std::unique_ptr<QnnTensorHolder>> prefill_output_holders_;
  std::vector<std::unique_ptr<QnnTensorHolder>> kv_input_holders_;
  std::vector<std::unique_ptr<QnnTensorHolder>> kv_output_holders_;
  
  // Helper methods (single-context)
  bool load_graphs();
  bool extract_metadata();
  bool setup_kv_cache();
  bool setup_io_allocators();
  
  // Helper methods (multi-context)
  bool load_multi_context_graphs();
  bool extract_multi_context_metadata();
  bool setup_multi_context_kv_cache();
  bool setup_multi_context_io_allocators();
  bool allocate_shared_buffers();

 public:
  // Single-context execution
  bool run_prefill(const std::vector<int32_t>& tokens, 
                   int32_t& next_token,
                   int32_t& n_update,
                   llama_context * llama_ctx = nullptr);
  
  bool run_decode_step(int32_t token_in,
                       int32_t n_past,
                       int32_t& token_out,
                       llama_context * llama_ctx = nullptr);
  
  // Multi-context execution (llama_batch interface)
  bool run_multi_context_prefill(llama_context * ctx, llama_batch batch);
  bool run_multi_context_decode_step(llama_context * ctx, llama_batch batch);
  
  /**
   * @brief llama_decode-compatible interface for QNN execution
   * @param ctx llama_context for logits injection
   * @param batch llama_batch containing tokens to process
   * @return 0 on success, non-zero on failure (matches llama_decode semantics)
   */
  int qnn_decode(llama_context * ctx, llama_batch batch);

 private:
  // Shard execution helpers
  bool run_shard_prefill(int shard_idx,
                         const std::vector<int32_t>& tokens,
                         int32_t n_past,
                         int32_t n_update);
  
  bool run_shard_decode(int shard_idx,
                        int32_t n_past);
};

} // namespace llama_qnn
