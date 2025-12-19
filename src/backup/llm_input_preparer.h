#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include "qnn_qnnjson.h"

namespace llama_qnn {

/**
 * @brief Prepares input tensors for LLM inference (tokens, positions, attention mask)
 */
class InputPreparer {
public:
  /**
   * @brief Fill token input tensor
   * @param buffer Destination buffer
   * @param tensor_desc Tensor descriptor
   * @param tokens Token IDs to fill
   * @return true if successful
   */
  static bool fill_tokens(
    void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    const std::vector<int32_t>& tokens);

  /**
   * @brief Fill position input tensor
   * @param buffer Destination buffer
   * @param tensor_desc Tensor descriptor
   * @param num_tokens Number of tokens
   * @param start_pos Starting position (default 0 for prefill)
   * @return true if successful
   */
  static bool fill_positions(
    void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    size_t num_tokens,
    int32_t start_pos = 0);

  /**
   * @brief Fill attention mask tensor (causal mask at context end)
   * @param buffer Destination buffer
   * @param tensor_desc Tensor descriptor
   * @param num_tokens Number of tokens
   * @return true if successful
   */
  static bool fill_attention_mask(
    void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    size_t num_tokens);

  /**
   * @brief Clear KV cache input tensors to 0
   * @param buffer Destination buffer
   * @param tensor_desc Tensor descriptor
   * @return true if successful
   */
  static bool clear_kv_cache(
    void* buffer,
    const QnnJsonTensorDesc& tensor_desc);

  /**
   * @brief Check if tensor is a KV cache tensor
   * @param tensor_desc Tensor descriptor
   * @return true if it's a KV cache tensor
   */
  static bool is_kv_cache_tensor(const QnnJsonTensorDesc& tensor_desc);

  /**
   * @brief Auto-fill input tensors based on heuristics
   * @param graph_desc Graph descriptor
   * @param get_buffer_fn Function to get buffer by tensor name
   * @param tokens Token IDs
   * @param verbose Print debug info
   * @return true if successful
   */
  static bool auto_fill_inputs(
    const QnnJsonGraphDesc& graph_desc,
    std::function<void*(const std::string&)> get_buffer_fn,
    const std::vector<int32_t>& tokens,
    bool verbose = true);
};

} // namespace llama_qnn

