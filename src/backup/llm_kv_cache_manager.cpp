#include "llm_kv_cache_manager.h"
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace llama_qnn {

LLMKVCacheManager::LLMKVCacheManager(const Metadata& metadata)
    : metadata_(metadata), total_cache_size_(0), cur_ar_len_(metadata.max_ar_len) {
  // Resize storage
  k_cache_.resize(metadata_.num_layers);
  v_cache_.resize(metadata_.num_layers);
  
  for (int32_t layer = 0; layer < metadata_.num_layers; ++layer) {
    k_cache_[layer].resize(metadata_.num_heads);
    v_cache_[layer].resize(metadata_.num_heads);
  }

  // Calculate total memory requirement (SMART_MASK mode)
  // Each cache: input_buffer + output_buffer
  size_t k_in_bytes = metadata_.head_dim * metadata_.max_cache_len;
  size_t k_out_bytes = metadata_.head_dim * metadata_.max_ar_len;
  size_t v_in_bytes = metadata_.head_dim * metadata_.max_cache_len;
  size_t v_out_bytes = metadata_.head_dim * metadata_.max_ar_len;
  
  size_t per_head = k_in_bytes + k_out_bytes + v_in_bytes + v_out_bytes;
  total_cache_size_ = per_head * metadata_.num_layers * metadata_.num_heads;
  
  std::cout << "[LLMKVCacheManager] Metadata:\n"
            << "  context_len: " << metadata_.context_len << "\n"
            << "  head_dim: " << metadata_.head_dim << "\n"
            << "  max_ar_len: " << metadata_.max_ar_len << "\n"
            << "  max_cache_len: " << metadata_.max_cache_len << "\n"
            << "  num_heads: " << metadata_.num_heads << "\n"
            << "  num_layers: " << metadata_.num_layers << "\n"
            << "  Total cache size: " << (total_cache_size_ / 1024.0 / 1024.0) << " MiB\n";
}

LLMKVCacheManager::~LLMKVCacheManager() {
  // Free all allocated memory
  for (auto& layer_k : k_cache_) {
    for (auto& head_k : layer_k) {
      if (head_k.input_buffer) free(head_k.input_buffer);
      if (head_k.output_buffer) free(head_k.output_buffer);
    }
  }
  for (auto& layer_v : v_cache_) {
    for (auto& head_v : layer_v) {
      if (head_v.input_buffer) free(head_v.input_buffer);
      if (head_v.output_buffer) free(head_v.output_buffer);
    }
  }
}

bool LLMKVCacheManager::allocate() {
  size_t k_in_bytes = metadata_.head_dim * metadata_.max_cache_len;
  size_t k_out_bytes = metadata_.head_dim * metadata_.max_ar_len;
  size_t v_in_bytes = metadata_.head_dim * metadata_.max_cache_len;
  size_t v_out_bytes = metadata_.head_dim * metadata_.max_ar_len;

  std::cout << "[LLMKVCacheManager] Allocating memory...\n";
  
  for (int32_t layer = 0; layer < metadata_.num_layers; ++layer) {
    for (int32_t head = 0; head < metadata_.num_heads; ++head) {
      // Allocate K cache
      k_cache_[layer][head].input_buffer = malloc(k_in_bytes);
      k_cache_[layer][head].output_buffer = malloc(k_out_bytes);
      k_cache_[layer][head].input_bytes = k_in_bytes;
      k_cache_[layer][head].output_bytes = k_out_bytes;
      
      if (!k_cache_[layer][head].input_buffer || !k_cache_[layer][head].output_buffer) {
        std::cerr << "[LLMKVCacheManager] Failed to allocate K cache for layer " 
                  << layer << ", head " << head << "\n";
        return false;
      }
      
      // Initialize to zero
      std::memset(k_cache_[layer][head].input_buffer, 0, k_in_bytes);
      std::memset(k_cache_[layer][head].output_buffer, 0, k_out_bytes);
      
      // Allocate V cache
      v_cache_[layer][head].input_buffer = malloc(v_in_bytes);
      v_cache_[layer][head].output_buffer = malloc(v_out_bytes);
      v_cache_[layer][head].input_bytes = v_in_bytes;
      v_cache_[layer][head].output_bytes = v_out_bytes;
      
      if (!v_cache_[layer][head].input_buffer || !v_cache_[layer][head].output_buffer) {
        std::cerr << "[LLMKVCacheManager] Failed to allocate V cache for layer " 
                  << layer << ", head " << head << "\n";
        return false;
      }
      
      // Initialize to zero
      std::memset(v_cache_[layer][head].input_buffer, 0, v_in_bytes);
      std::memset(v_cache_[layer][head].output_buffer, 0, v_out_bytes);
    }
  }
  
  std::cout << "[LLMKVCacheManager] Allocation complete: " 
            << (total_cache_size_ / 1024.0 / 1024.0) << " MiB\n";
  return true;
}

void LLMKVCacheManager::update_key_cache(
    const KVCacheBuffer& cache,
    int32_t n_past,
    int32_t n_update) {
  // Key cache layout (SMART_MASK):
  // Input:  [head_dim, max_cache_len]
  // Output: [head_dim, max_ar_len]
  //
  // Update: For each dimension, copy output[dim][0:n_update] → input[dim][n_past:n_past+n_update]
  
  uint8_t* write_ptr = reinterpret_cast<uint8_t*>(cache.input_buffer) + n_past;
  uint8_t* read_ptr = reinterpret_cast<uint8_t*>(cache.output_buffer);
  
  for (int32_t dim = 0; dim < metadata_.head_dim; ++dim) {
    std::memcpy(write_ptr, read_ptr, n_update);
    write_ptr += metadata_.max_cache_len;
    read_ptr += metadata_.max_ar_len;
  }
}

void LLMKVCacheManager::update_value_cache(
    const KVCacheBuffer& cache,
    int32_t n_past,
    int32_t n_update) {
  // Value cache layout (SMART_MASK):
  // Input:  [max_cache_len, head_dim] - sequential
  // Output: [max_ar_len, head_dim] - sequential
  //
  // Update: copy output[0:n_update*head_dim] → input[n_past*head_dim:(n_past+n_update)*head_dim]
  
  uint8_t* write_ptr = reinterpret_cast<uint8_t*>(cache.input_buffer) + n_past * metadata_.head_dim;
  uint8_t* read_ptr = reinterpret_cast<uint8_t*>(cache.output_buffer);
  
  std::memcpy(write_ptr, read_ptr, n_update * metadata_.head_dim);
}

void LLMKVCacheManager::update_cache(int32_t n_past, int32_t n_update) {
  for (int32_t layer = 0; layer < metadata_.num_layers; ++layer) {
    for (int32_t head = 0; head < metadata_.num_heads; ++head) {
      update_key_cache(k_cache_[layer][head], n_past, n_update);
      update_value_cache(v_cache_[layer][head], n_past, n_update);
    }
  }
}

void LLMKVCacheManager::init_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past) {
  // SMART_MASK attention mask initialization
  // Shape: [ar_len, context_len]
  // Values: 0 = mask (don't attend), 65535 = attend
  //
  // Pattern: Causal mask at the END of context window
  // For prefill (ar_len > 1):
  //   Row i attends to [context_len - ar_len, context_len - ar_len + i]
  // For decode (ar_len = 1):
  //   Row 0 attends to [0, n_past]
  
  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  
  // Clear all to 0 (mask)
  std::memset(attention_mask, 0, ar_len * metadata_.context_len * sizeof(uint16_t));
  
  for (int32_t i = 0; i < ar_len; ++i) {
    uint16_t* row = attention_mask + i * metadata_.context_len;
    
    // Attend to all past tokens (before this batch)
    for (int32_t j = 0; j < n_past; ++j) {
      row[j] = pos_val;
    }
    
    // Attend to tokens in current batch (causal)
    // New tokens are placed at context window end: [context_len - ar_len, context_len)
    int32_t new_token_start = metadata_.context_len - ar_len;
    for (int32_t j = 0; j <= i; ++j) {
      row[new_token_start + j] = pos_val;
    }
  }
}

void LLMKVCacheManager::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  // SMART_MASK attention mask update
  // After cache update, newly added tokens should be attended to
  //
  // Update pattern: For each row, fill [n_past, n_past + n_update) with 65535
  
  uint16_t pos_val = 65535;
  
  for (int32_t i = 0; i < ar_len; ++i) {
    uint16_t* row = attention_mask + i * metadata_.context_len;
    std::fill_n(row + n_past, n_update, pos_val);
  }
}

void LLMKVCacheManager::rearrange_key(
    KVCacheBuffer& cache,
    int32_t src_cache_len,
    int32_t dst_cache_len) {
  // ExecutorchReader의 rearrange_key 구현:
  // 
  // K cache layout: [head_dim, cache_len] - strided
  // Prefill→Decode 전환 시 [64, 480] → [64, 511]로 확장
  // 
  // 각 dimension 별로 memmove:
  //   src: [dim][0..479]
  //   dst: [dim][0..479] (31 positions shifted)
  // 
  // 예: head_dim=64
  //   Before: [dim0][0..479][dim1][0..479]...[dim63][0..479]
  //   After:  [dim0][0..510][dim1][0..510]...[dim63][0..510]
  //   
  //   memmove(&cache[dim*511], &cache[dim*480], 480)로 각 dim을 이동
  
  if (src_cache_len == dst_cache_len) {
    return;  // No rearrangement needed
  }
  
  // std::cout << "[LLMKVCacheManager] Rearranging K cache: " 
  //           << src_cache_len << " → " << dst_cache_len << "\n";
  
  uint8_t* buffer = reinterpret_cast<uint8_t*>(cache.input_buffer);
  
  // // DEBUG: Log BEFORE rearrange (first cache only, Layer 0 Head 0)
  // static bool first_rearrange = true;
  // if (first_rearrange && cache.input_buffer == k_cache_[0][0].input_buffer) {
  //   std::cout << "[DEBUG Rearrange] BEFORE K cache (L0H0):\n";
  //   std::cout << "  buffer[0]=" << (int)buffer[0] 
  //             << ", buffer[" << src_cache_len << "]=" << (int)buffer[src_cache_len]
  //             << ", buffer[" << (src_cache_len*2) << "]=" << (int)buffer[src_cache_len*2] << "\n";
  //   std::cout << "  Total buffer size allocated: " << cache.input_bytes << " bytes\n";
  //   std::cout << "  src_cache_len=" << src_cache_len << ", dst_cache_len=" << dst_cache_len << "\n";
  // }
  
  // BACKWARD iteration to avoid overwrite (src_cache_len < dst_cache_len)
  for (int32_t dim = metadata_.head_dim - 1; dim >= 0; --dim) {
    uint8_t* src = buffer + dim * src_cache_len;
    uint8_t* dst = buffer + dim * dst_cache_len;
    std::memmove(dst, src, src_cache_len);
  }
  
  // // DEBUG: Log AFTER rearrange
  // if (first_rearrange && cache.input_buffer == k_cache_[0][0].input_buffer) {
  //   std::cout << "[DEBUG Rearrange] AFTER K cache (L0H0):\n";
  //   std::cout << "  buffer[0]=" << (int)buffer[0] 
  //             << ", buffer[" << dst_cache_len << "]=" << (int)buffer[dst_cache_len]
  //             << ", buffer[" << (dst_cache_len*2) << "]=" << (int)buffer[dst_cache_len*2] << "\n";
    
  //   // 전체 non-zero count
  //   int non_zero = 0;
  //   for (int i = 0; i < std::min((int)cache.input_bytes, 10000); ++i) {
  //     if (buffer[i] != 0) non_zero++;
  //   }
  //   std::cout << "  Non-zero in first 10000 bytes: " << non_zero << "/10000\n";
  //   first_rearrange = false;
  // }
}

void LLMKVCacheManager::rearrange_value(
    KVCacheBuffer& cache,
    int32_t src_cache_len,
    int32_t dst_cache_len) {
  // ExecutorchReader의 rearrange_value 구현:
  // 
  // V cache layout: [cache_len, head_dim] - sequential
  // Prefill→Decode 전환 시 [480, 64] → [511, 64]로 확장
  // 
  // Sequential이므로 단순 copy (이미 올바른 위치):
  //   [0..479*64] → [0..479*64] (no change)
  //   [480*64..510*64] → all zeros (new positions)
  // 
  // ExecutorchReader는 실제로 아무것도 안함 (already correct)
  
  if (src_cache_len == dst_cache_len) {
    return;  // No rearrangement needed
  }
  
  // std::cout << "[LLMKVCacheManager] Rearranging V cache: " 
  //           << src_cache_len << " → " << dst_cache_len << " (no-op)\n";
  
  // V cache is sequential, so no rearrangement needed
  // The first src_cache_len * head_dim bytes are already in correct position
}

void LLMKVCacheManager::rearrange_cache(int32_t src_ar_len, int32_t dst_ar_len) {
  // ExecutorchReader의 rearrange_cache 구현:
  // 
  // Prefill (AR=32) → Decode (AR=1) 전환 시 호출
  // - Prefill cache_len: context_len - 32 = 512 - 32 = 480
  // - Decode cache_len:  context_len - 1  = 512 - 1  = 511
  // 
  // 모든 layer/head의 K/V cache를 480→511로 재배치
  
  if (src_ar_len == dst_ar_len) {
    std::cout << "[LLMKVCacheManager] Rearrange skipped (same AR len: " 
              << src_ar_len << ")\n";
    return;
  }
  
  int32_t src_cache_len = metadata_.context_len - src_ar_len;
  int32_t dst_cache_len = metadata_.context_len - dst_ar_len;
  
  std::cout << "[LLMKVCacheManager] Rearranging cache: AR " 
            << src_ar_len << " → " << dst_ar_len 
            << " (cache " << src_cache_len << " → " << dst_cache_len << ")\n";
  
  for (int32_t layer = 0; layer < metadata_.num_layers; ++layer) {
    for (int32_t head = 0; head < metadata_.num_heads; ++head) {
      rearrange_key(k_cache_[layer][head], src_cache_len, dst_cache_len);
      rearrange_value(v_cache_[layer][head], src_cache_len, dst_cache_len);
    }
  }
  
  cur_ar_len_ = dst_ar_len;  // Update current AR length
  std::cout << "[LLMKVCacheManager] Rearrange complete\n";
}

} // namespace llama_qnn

