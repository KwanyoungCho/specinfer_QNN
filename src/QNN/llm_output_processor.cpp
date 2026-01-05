#include "llm_output_processor.h"
#include <iostream>

namespace llama_qnn {

bool OutputProcessor::dequantize_logits(
    const void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    std::vector<float>& output_floats) {
  if (!buffer) return false;
  
  // Check if it's UFIXED_POINT_16
  if (tensor_desc.data_type.find("UFIXED_POINT_16") == std::string::npos) {
    std::cerr << "[OutputProcessor] Expected UFIXED_POINT_16, got " 
              << tensor_desc.data_type << "\n";
    return false;
  }
  
  // Get quantization params
  float scale = tensor_desc.quant_scale;
  int32_t offset = tensor_desc.quant_offset;
  
  if (scale == 0.0f) {
    std::cerr << "[OutputProcessor] WARNING: quant_scale is 0.0! Cannot dequantize.\n";
    std::cerr << "[OutputProcessor] tensor: " << tensor_desc.name 
              << ", encoding: " << tensor_desc.quant_encoding << "\n";
    return false;
  }
  
  // Calculate total elements
  size_t total_elements = 1;
  for (uint64_t d : tensor_desc.dims) {
    total_elements *= d;
  }
  
  output_floats.resize(total_elements);
  const uint16_t* quant_data = reinterpret_cast<const uint16_t*>(buffer);
  
  for (size_t i = 0; i < total_elements; ++i) {
    output_floats[i] = (static_cast<float>(quant_data[i]) + offset) * scale;
  }
  
  return true;
}

} // namespace llama_qnn

