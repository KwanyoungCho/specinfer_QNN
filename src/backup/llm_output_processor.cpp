#include "llm_output_processor.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

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

int32_t OutputProcessor::argmax(const std::vector<float>& logits, size_t offset) {
  if (logits.empty()) return -1;
  
  // For LLM logits: offset is row start, search from offset to end (one row)
  size_t vocab_size = logits.size() - offset;
  if (offset >= logits.size()) return -1;
  
  float max_val = logits[offset];
  int32_t max_vocab_id = 0; // Return vocab ID, not absolute index
  
  for (size_t i = 1; i < vocab_size; ++i) {
    size_t abs_idx = offset + i;
    if (abs_idx >= logits.size()) break;
    
    if (logits[abs_idx] > max_val) {
      max_val = logits[abs_idx];
      max_vocab_id = static_cast<int32_t>(i);
    }
  }
  
  return max_vocab_id;
}

void OutputProcessor::topk(
    const std::vector<float>& logits,
    size_t k,
    size_t offset,
    std::vector<int32_t>& top_indices,
    std::vector<float>& top_values) {
  
  if (logits.empty()) return;
  
  // For LLM logits: offset is the row start (seq_pos * vocab_size)
  // We want to find topk within a single row (vocab_size elements)
  // Calculate vocab size assuming logits contains full output
  size_t vocab_size = logits.size(); // For single token, this is vocab_size
  size_t start_idx = offset;
  
  // If offset suggests multi-token output, calculate vocab size
  if (offset > 0 && offset < logits.size()) {
    // Assume square-ish layout or deduce from offset
    // For safety, search from offset to end
    vocab_size = logits.size() - offset;
  }
  
  // Create pairs of (vocab_id, value) for sorting
  std::vector<std::pair<int32_t, float>> idx_val_pairs;
  for (size_t i = 0; i < vocab_size && (start_idx + i) < logits.size(); ++i) {
    idx_val_pairs.push_back({static_cast<int32_t>(i), logits[start_idx + i]});
  }
  
  // Partial sort to get top k
  size_t actual_k = std::min(k, idx_val_pairs.size());
  std::partial_sort(
    idx_val_pairs.begin(),
    idx_val_pairs.begin() + actual_k,
    idx_val_pairs.end(),
    [](const auto& a, const auto& b) { return a.second > b.second; }
  );
  
  top_indices.clear();
  top_values.clear();
  for (size_t i = 0; i < actual_k; ++i) {
    top_indices.push_back(idx_val_pairs[i].first);
    top_values.push_back(idx_val_pairs[i].second);
  }
}

void OutputProcessor::print_topk(
    const std::vector<float>& logits,
    size_t k,
    size_t offset) {
  
  std::vector<int32_t> indices;
  std::vector<float> values;
  topk(logits, k, offset, indices, values);
  
  std::cout << "top-" << k << " logits (id:val): ";
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) std::cout << " ";
    std::cout << indices[i] << ":" << std::fixed << std::setprecision(2) << values[i];
  }
  std::cout << "\n";
}

int32_t OutputProcessor::dequantize_and_argmax(
    const void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    size_t offset,
    std::vector<float>* output_logits) {
  
  std::vector<float> logits;
  if (!dequantize_logits(buffer, tensor_desc, logits)) {
    return -1;
  }
  
  if (output_logits) {
    *output_logits = std::move(logits);
    return argmax(*output_logits, offset);
  } else {
    return argmax(logits, offset);
  }
}

} // namespace llama_qnn

