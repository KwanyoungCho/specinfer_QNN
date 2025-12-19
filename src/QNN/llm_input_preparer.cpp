#include "llm_input_preparer.h"
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cctype>
#include <iostream>

namespace llama_qnn {

bool InputPreparer::is_kv_cache_tensor(const QnnJsonTensorDesc& tensor_desc) {
  std::string name_lower = tensor_desc.name;
  for (auto& c : name_lower) c = (char)tolower(c);
  
  // Pattern: input_N_args_M_0 with 3D shape and UFIXED_POINT_8
  bool has_args = name_lower.find("_args_") != std::string::npos;
  bool is_3d = tensor_desc.dims.size() == 3;
  bool is_u8 = tensor_desc.data_type.find("UFIXED_POINT_8") != std::string::npos;
  
  return has_args && is_3d && is_u8;
}

bool InputPreparer::fill_tokens(
    void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    const std::vector<int32_t>& tokens) {
  if (!buffer || tokens.empty()) return false;
  
  size_t elem_bytes = 4; // INT_32/UINT_32
  size_t need = tokens.size() * elem_bytes;
  
  if (need > tensor_desc.nbytes) {
    std::cerr << "[InputPreparer] Token buffer too small: need " 
              << need << ", have " << tensor_desc.nbytes << "\n";
    return false;
  }
  
  std::memcpy(buffer, tokens.data(), need);
  return true;
}

bool InputPreparer::fill_positions( // [spagetti] fill token이랑 fill position, (mask까지)이랑 왜 데이터 집어 넣는 방식이 달라?이유가 있긴한데 일단 좀 깔끔하게 고치긴 했습니다. 
    void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    size_t num_tokens,
    int32_t start_pos) {
  if (!buffer || num_tokens == 0) return false;
  
  if (tensor_desc.nbytes < num_tokens * sizeof(int32_t)) {
    std::cerr << "[InputPreparer] Position buffer too small\n";
    return false;
  }
  
  int32_t* pos_buf = reinterpret_cast<int32_t*>(buffer);
  std::iota(pos_buf, pos_buf + num_tokens, start_pos);
  
  return true;
}

bool InputPreparer::fill_attention_mask(
    void* buffer,
    const QnnJsonTensorDesc& tensor_desc,
    size_t num_tokens) {
  if (!buffer || num_tokens == 0) return false;
  
  if (tensor_desc.dims.size() < 2) return false;
  
  uint64_t seq_dim = tensor_desc.dims[tensor_desc.dims.size() - 2];
  uint64_t max_len = tensor_desc.dims.back();
  
  if (tensor_desc.data_type.find("UFIXED_POINT_16") == std::string::npos) {
    return false;
  }
  
  uint16_t* mask_buf = reinterpret_cast<uint16_t*>(buffer);
  uint64_t fill_seq = std::min<uint64_t>(num_tokens, seq_dim);
  
  // Clear all to 0 (masked = cannot attend)
  std::memset(mask_buf, 0, tensor_desc.nbytes);
  
  // SMART_MASK 방식: Prefill 초기화 (n_past=0인 경우)
  // 
  // Context window 구조:
  // [0 .. max_len-seq_dim-1] : Past KV cache 영역 (현재는 비어있음, n_past=0)
  // [max_len-seq_dim .. max_len-1] : New tokens 영역 (여기에 현재 토큰들 배치)
  //
  // Example: seq_dim=32, max_len=512, num_tokens=12
  // - Past 영역: [0..479] (비어있음)
  // - New tokens: [480..511] (여기에 12개 토큰 배치)
  //
  // Attention mask:
  // Row 0:  [0, 0, ..., 0, 65535, 0, 0, ...]         ← position 480에만 attend
  // Row 1:  [0, 0, ..., 0, 65535, 65535, 0, ...]     ← position 480-481에 attend (causal)
  // Row 11: [0, 0, ..., 0, 65535, ..., 65535, ...]   ← position 480-491에 attend
  // Row 12-31: [0, 0, ..., 0]                        ← 토큰 없음 (나중에 업데이트됨)
  
  for (uint64_t i = 0; i < fill_seq; ++i) {
    uint64_t row_offset = i * max_len;
    uint64_t attend_start = max_len - seq_dim; // New tokens의 시작 위치
    
    // Causal mask: token i는 자기 자신(i)과 이전 토큰들(0..i-1)에만 attend
    std::fill_n(mask_buf + row_offset + attend_start, i + 1, 65535);
  }
  
  // Note: fill_seq < seq_dim인 경우, 나머지 행들은 0으로 유지
  // 이후 prefill iteration이나 decode에서 update_attention_mask로 채워짐
  
  return true;
}

bool InputPreparer::clear_kv_cache(
    void* buffer,
    const QnnJsonTensorDesc& tensor_desc) {
  if (!buffer) return false;
  std::memset(buffer, 0, tensor_desc.nbytes);
  return true;
}

bool InputPreparer::auto_fill_inputs(
    const QnnJsonGraphDesc& graph_desc,
    std::function<void*(const std::string&)> get_buffer_fn,
    const std::vector<int32_t>& tokens,
    int32_t start_pos,
    bool skip_attention_mask,
    bool verbose) {
  
  for (const auto& t : graph_desc.inputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    
    void* buffer = get_buffer_fn(t.name);
    if (!buffer) continue;
    
    // Token input
    bool is_token_input = name_lower.find("token") != std::string::npos;
    bool is_int32 = t.data_type.find("INT_32") != std::string::npos || 
                    t.data_type.find("UINT_32") != std::string::npos;
    bool is_1d_or_2d = t.dims.size() == 1 || t.dims.size() == 2;
    
    if (is_token_input && is_int32 && is_1d_or_2d) {
      if (fill_tokens(buffer, t, tokens)) {
        if (verbose) {
          std::cout << "[InputPreparer] Filled tokens: " << t.name 
                    << " count=" << tokens.size() << "\n";
        }
      }
      continue;
    }
    
    // Position input
    bool is_position = name_lower.find("_pos_") != std::string::npos;
    
    if (is_position && is_int32 && t.nbytes >= tokens.size() * 4) {
      if (fill_positions(buffer, t, tokens.size(), start_pos)) {
        if (verbose) {
          std::cout << "[InputPreparer] Filled positions: " << t.name 
                    << " " << start_pos << ".." << (start_pos + tokens.size() - 1) << "\n";
        }
      }
      continue;
    }
    
    // Attention mask (skip if requested - for multi-iteration prefill)
    bool is_mask = name_lower.find("atten_mask") != std::string::npos;
    
    if (is_mask && t.dims.size() >= 2 && !skip_attention_mask) {
      if (fill_attention_mask(buffer, t, tokens.size())) {
        if (verbose) {
          std::cout << "[InputPreparer] Filled attention mask: " << t.name 
                    << " tokens=" << tokens.size() << "\n";
        }
      }
      continue;
    }
  }
  
  return true;
}

} // namespace llama_qnn

