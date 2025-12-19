#include "llm_kv_cache_mapper.h"
#include <algorithm>

namespace llama_qnn {

int LLMKVCacheMapper::extract_input_index(const std::string& name) {
  size_t pos = name.find("input_");
  if (pos == std::string::npos) return -1;
  
  size_t start = pos + 6;  // "input_" length
  size_t end = name.find('_', start);
  if (end == std::string::npos) return -1;
  
  try {
    return std::stoi(name.substr(start, end - start));
  } catch (...) {
    return -1;
  }
}

std::vector<KVCacheTensorInfo> LLMKVCacheMapper::build_mapping(
    const QnnJsonGraphDesc& graph,
    int num_heads,
    int head_dim) {
  
  std::vector<KVCacheTensorInfo> result;
  
  // Collect all 3D tensors with input_X pattern
  for (const auto& t : graph.inputs) {
    if (t.dims.size() != 3) continue;
    
    int input_idx = extract_input_index(t.name);
    if (input_idx < 2) continue;  // Skip tokens (0), pos (1)
    
    // Detect V vs K by shape
    // V cache: [1, cache_len, head_dim] where cache_len > head_dim
    // K cache: [1, head_dim, cache_len] where cache_len > head_dim
    bool is_v = (t.dims[1] > head_dim && t.dims[2] == head_dim);
    bool is_k = (t.dims[1] == head_dim && t.dims[2] > head_dim);
    
    if (is_v || is_k) {
      KVCacheTensorInfo info;
      info.name = t.name;
      info.input_idx = input_idx;
      info.layer = -1;
      info.head = -1;
      info.is_v_cache = is_v;
      info.dims = t.dims;
      result.push_back(info);
    }
  }
  
  // Sort by input_idx
  std::sort(result.begin(), result.end(), 
            [](const KVCacheTensorInfo& a, const KVCacheTensorInfo& b) {
              return a.input_idx < b.input_idx;
            });
  
  // Assign layer/head based on pattern: V 8개 → K 8개 per layer
  int v_count = 0, k_count = 0;
  for (auto& info : result) {
    if (info.is_v_cache) {
      info.layer = v_count / num_heads;
      info.head = v_count % num_heads;
      v_count++;
    } else {
      info.layer = k_count / num_heads;
      info.head = k_count % num_heads;
      k_count++;
    }
  }
  
  return result;
}

std::map<std::string, void*> LLMKVCacheMapper::create_buffer_override(
    const std::vector<KVCacheTensorInfo>& mapping,
    LLMKVCacheManager& kv_manager) {
  
  std::map<std::string, void*> override_map;
  
  for (const auto& info : mapping) {
    if (info.is_v_cache) {
      const auto& v_buf = kv_manager.get_v_cache(info.layer, info.head);
      override_map[info.name] = v_buf.input_buffer;
    } else {
      const auto& k_buf = kv_manager.get_k_cache(info.layer, info.head);
      override_map[info.name] = k_buf.input_buffer;
    }
  }
  
  return override_map;
}

} // namespace llama_qnn
