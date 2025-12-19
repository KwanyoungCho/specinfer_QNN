#pragma once

#include "qnn_qnnjson.h"
#include "llm_kv_cache_manager.h"
#include <map>
#include <string>
#include <vector>

namespace llama_qnn {

/**
 * @brief KV cache tensor mapping info
 */
struct KVCacheTensorInfo {
  std::string name;
  int input_idx;      // Input tensor index (e.g., input_2 → 2)
  int layer;
  int head;
  bool is_v_cache;    // true: V cache, false: K cache
  std::vector<uint32_t> dims;
};

/**
 * @brief Maps QNN JSON KV cache tensors to LLMKVCacheManager buffers
 * 
 * ExecutorchReader pattern analysis:
 *   - prefill_forward/kv_forward both follow:
 *     input_2~9:   V cache L0 H0~7 [1, cache_len, 64]
 *     input_10~17: K cache L0 H0~7 [1, 64, cache_len]
 *     input_18:    attention_mask
 *     input_19~26: V cache L1 H0~7
 *     input_27~34: K cache L1 H0~7
 *     ... repeated per layer ...
 *   
 *   Pattern: Each layer has V 8개 → K 8개
 */
class LLMKVCacheMapper {
 public:
  /**
   * @brief Build KV cache mapping from QNN JSON graph
   * @param graph QNN graph description
   * @param num_heads Number of attention heads
   * @param head_dim Dimension per head
   * @return KV tensor mapping info
   */
  static std::vector<KVCacheTensorInfo> build_mapping(
      const QnnJsonGraphDesc& graph,
      int num_heads,
      int head_dim);
  
  /**
   * @brief Create buffer override map for KV cache inputs
   * @param mapping KV tensor mapping info
   * @param kv_manager KV cache manager with allocated buffers
   * @return Map of tensor_name → buffer pointer
   */
  static std::map<std::string, void*> create_buffer_override(
      const std::vector<KVCacheTensorInfo>& mapping,
      LLMKVCacheManager& kv_manager);
  
 private:
  static int extract_input_index(const std::string& name);
};

} // namespace llama_qnn
