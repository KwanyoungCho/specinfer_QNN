#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "qnn_qnnjson.h"

namespace llama_qnn {

/**
 * @brief Processes LLM output tensors (logits dequantization, argmax, topk)
 */
class OutputProcessor {
public:
  /**
   * @brief Dequantize UFIXED_POINT_16 logits to float
   * @param buffer Source buffer (quantized logits)
   * @param tensor_desc Tensor descriptor with quantization params
   * @param output_floats Destination vector for float values
   * @return true if successful
   */
  static bool dequantize_logits(
    const void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    std::vector<float>& output_floats);

  /**
   * @brief Find argmax from logits
   * @param logits Float logits
   * @param offset Offset to start searching (for multi-token outputs)
   * @return Index of maximum value
   */
  static int32_t argmax(const std::vector<float>& logits, size_t offset = 0);

  /**
   * @brief Get top-K indices and values
   * @param logits Float logits
   * @param k Number of top values to return
   * @param offset Offset to start searching
   * @param top_indices Output vector for indices
   * @param top_values Output vector for values
   */
  static void topk(
    const std::vector<float>& logits,
    size_t k,
    size_t offset,
    std::vector<int32_t>& top_indices,
    std::vector<float>& top_values);

  /**
   * @brief Print top-K logits
   * @param logits Float logits
   * @param k Number of top values to print
   * @param offset Offset to start searching
   */
  static void print_topk(
    const std::vector<float>& logits,
    size_t k = 10,
    size_t offset = 0);

  /**
   * @brief Dequantize and get argmax in one call
   * @param buffer Quantized logits buffer
   * @param tensor_desc Tensor descriptor
   * @param offset Offset for argmax
   * @param output_logits Optional: store dequantized logits
   * @return Argmax token ID, or -1 on error
   */
  static int32_t dequantize_and_argmax(
    const void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    size_t offset = 0,
    std::vector<float>* output_logits = nullptr);
};

} // namespace llama_qnn

