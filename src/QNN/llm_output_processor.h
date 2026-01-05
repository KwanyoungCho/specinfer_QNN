#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "qnn_qnnjson.h"

namespace llama_qnn {

/**
 * @brief Processes LLM output tensors (logits dequantization)
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
};

} // namespace llama_qnn

