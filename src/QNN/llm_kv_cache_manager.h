#pragma once

#include <cstdint>
#include <set>
#include <vector>
#include <memory>
#include <string>

namespace llama_qnn {

/**
 * @brief LLM KV Cache Manager for SMART_MASK mode
 * 
 * Manages KV cache memory allocation and updates for dual-graph LLM inference.
 * Compatible with ExecuTorch's KVManager design.
 * 
 * Key Responsibilities:
 * - Allocate persistent KV cache buffers (input + output)
 * - Rearrange KV cache layout on prefill→decode transition
 * - Track KV cell metadata for speculative decoding (seq_rm/keep/cp)
 * - Provide memory pointers for graph binding
 */
class LLMKVCacheManager {
public:
  struct Metadata {
    int32_t context_len;      // Total context length (e.g., 256)
    int32_t head_dim;         // Attention head dimension (e.g., 64)
    int32_t max_ar_len;       // Max autoregressive length (e.g., 32 for prefill)
    int32_t max_cache_len;    // Max cache length (context_len - min_ar_len, e.g., 224)
    int32_t num_heads;        // Number of attention heads per layer (e.g., 8)
    int32_t num_layers;       // Number of transformer layers (e.g., 16)
  };

  /**
   * @brief KV cache buffer pair for one head
   */
  struct KVCacheBuffer {
    void* input_buffer;   // Persistent storage [head_dim, max_cache_len]
    void* output_buffer;  // Temporary storage [head_dim, max_ar_len]
    size_t input_bytes;
    size_t output_bytes;
  };

  /**
   * @brief KV cache cell metadata for tracking position and sequences
   * Supports multiple sequences per cell (like llama_kv_cells)
   */
  struct KVCellMeta {
    int32_t pos = -1;              // -1 = empty/invalid, >= 0 = valid position
    std::set<int32_t> seq;         // set of sequence IDs this cell belongs to
    
    bool is_empty() const { return pos == -1; }
    bool has_seq(int32_t seq_id) const { return seq.count(seq_id) > 0; }
    int seq_count() const { return static_cast<int>(seq.size()); }
  };

  LLMKVCacheManager(const Metadata& metadata);
  ~LLMKVCacheManager();

  /**
   * @brief Allocate all KV cache memory
   * @return true if successful
   */
  bool allocate();

  /**
   * @brief Get K cache buffer for a specific layer and head
   */
  const KVCacheBuffer& get_k_cache(int32_t layer, int32_t head) const {
    return k_cache_[layer][head];
  }

  /**
   * @brief Get V cache buffer for a specific layer and head
   */
  const KVCacheBuffer& get_v_cache(int32_t layer, int32_t head) const {
    return v_cache_[layer][head];
  }

  /**
   * @brief Get total allocated memory size
   */
  size_t total_cache_size() const { return total_cache_size_; }

  /**
   * @brief Get metadata
   */
  const Metadata& metadata() const { return metadata_; }
  
  /**
   * @brief Rearrange KV cache from src_ar_len layout to dst_ar_len layout
   * 
   * ExecutorchReader의 rearrange_cache 구현:
   * - Prefill (AR=32): cache_len = context_len - 32 = 480
   * - Decode (AR=1):   cache_len = context_len - 1  = 511
   * 
   * Prefill→Decode 전환 시 480→511로 메모리 재배치 필요
   * 
   * @param src_ar_len Source AR length (e.g., 32 for prefill)
   * @param dst_ar_len Destination AR length (e.g., 1 for decode)
   */
  void rearrange_cache(int32_t src_ar_len, int32_t dst_ar_len);
  
  /**
   * @brief Get current cache length for given AR length
   */
  int32_t get_cache_len_for_ar(int32_t ar_len) const {
    return metadata_.context_len - ar_len;
  }

  /**
   * @brief Build prefill attention mask (2D: [ar_len, context_len])
   *
   * Supports tree attention: causal + seq_id intersection check.
   * @param attn_mask  Output buffer, caller must allocate ar_len * context_len uint16_t
   * @param ar_len     Prefill AR length (e.g. prefill_ar_len_)
   * @param cache_len  Past KV cache length (e.g. prefill_cache_len_)
   * @param n_tokens   Actual tokens in this chunk (<= ar_len)
   * @param positions  positions[n_tokens] — batch.pos + processed, nullable (sequential fallback)
   * @param n_seq_ids  n_seq_ids[n_tokens] — batch.n_seq_id + processed, nullable
   * @param seq_ids    seq_ids[n_tokens]   — batch.seq_id + processed, nullable
   */
  void build_prefill_attn_mask(
      uint16_t*       attn_mask,
      int32_t         ar_len,
      int32_t         cache_len,
      int32_t         n_tokens,
      const int32_t*  positions,
      const int32_t*  n_seq_ids,
      int32_t* const* seq_ids) const;

  /**
   * @brief Build decode attention mask (1D: [context_len])
   *
   * Attends to all non-empty past KV cells + the last position (current token).
   * @param attn_mask  Output buffer, caller must allocate context_len uint16_t
   * @param cache_len  Past KV cache length (e.g. kv_cache_len_)
   */
  void build_decode_attn_mask(
      uint16_t* attn_mask,
      int32_t   cache_len) const;

  /**
   * @brief Write QNN KV output buffers back to KV input (persistent) buffers.
   *
   * Unified for prefill (n_tokens > 1) and decode (n_tokens == 1).
   * V layout: [cache_len, head_dim] — sequential, copy n_tokens rows.
   * K layout: [head_dim, cache_len] — strided, copy n_tokens columns per dim.
   *
   * @param v_outputs        V output pointers collected from shard bindings (order = layer*heads + head)
   * @param k_outputs        K output pointers collected from shard bindings
   * @param slot             Starting KV cell index (from find_slot)
   * @param n_tokens         Tokens to write (chunk_size for prefill, 1 for decode)
   * @param cache_len        Current KV input cache length (prefill_cache_len_ or kv_cache_len_)
   * @param ar_len           Current AR length (prefill_ar_len_ or kv_ar_len_)
   * @param shard_layer_base Global layer offset for this shard (shard_idx * layers_per_shard_)
   */
  void write_back_kv(
      const std::vector<void*>& v_outputs,
      const std::vector<void*>& k_outputs,
      int32_t slot,
      int32_t n_tokens,
      int32_t cache_len,
      int32_t ar_len,
      int32_t shard_layer_base);

  // ========== KV Cell Metadata API (llama_memory_seq_* compatible) ==========
  
  /**
   * @brief Remove seq_id from cells in range [p0, p1)
   * If p0 == -1, start from 0. If p1 == -1, go to end.
   * Cell becomes empty if no sequences remain.
   */
  void seq_rm(int32_t seq_id, int32_t p0, int32_t p1);
  
  /**
   * @brief Keep only cells that have seq_id, clear others
   */
  void seq_keep(int32_t seq_id);
  
  /**
   * @brief Copy src_seq to dst_seq for cells in range [p0, p1)
   * If p0 == -1, start from 0. If p1 == -1, go to end.
   */
  void seq_cp(int32_t src_seq, int32_t dst_seq, int32_t p0, int32_t p1);
  
  /**
   * @brief Find contiguous empty slots for n_tokens
   * @param n_tokens Number of tokens to find space for
   * @param hint Starting position hint (like head in llama.cpp)
   * @return Starting position, or -1 if not found
   */
  int32_t find_slot(int32_t n_tokens, int32_t hint = 0);
  
  /**
   * @brief Get number of used (non-empty) cells in KV cache
   */
  int32_t get_used_count() const;
  
  /**
   * @brief Check if a cell is empty
   */
  bool is_cell_empty(int32_t pos) const {
    return pos < 0 || pos >= (int32_t)cell_meta_.size() || cell_meta_[pos].is_empty();
  }
  
  /**
   * @brief Check if cell has specific sequence
   */
  bool cell_has_seq(int32_t pos, int32_t seq_id) const {
    if (pos < 0 || pos >= (int32_t)cell_meta_.size()) return false;
    return cell_meta_[pos].has_seq(seq_id);
  }
  
  /**
   * @brief Get cell metadata at position
   */
  const KVCellMeta& get_cell_meta(int32_t pos) const {
    return cell_meta_[pos];
  }
  
  /**
   * @brief Get all cell metadata (for attention mask generation)
   */
  const std::vector<KVCellMeta>& get_all_cell_meta() const {
    return cell_meta_;
  }
  
  /**
   * @brief Set cell position (for apply_ubatch pattern)
   */
  void set_cell_pos(int32_t cell_idx, int32_t pos) {
    if (cell_idx >= 0 && cell_idx < (int32_t)cell_meta_.size()) {
      cell_meta_[cell_idx].pos = pos;
    }
  }
  
  /**
   * @brief Add seq_id to cell (for apply_ubatch pattern)
   */
  void add_cell_seq(int32_t cell_idx, int32_t seq_id) {
    if (cell_idx >= 0 && cell_idx < (int32_t)cell_meta_.size() && seq_id >= 0) {
      cell_meta_[cell_idx].seq.insert(seq_id);
    }
  }
  
  /**
   * @brief Reset all cell metadata to empty
   */
  void reset_cell_meta();

private:
  Metadata metadata_;
  size_t total_cache_size_;

  // KV cache storage: [num_layers][num_heads]
  std::vector<std::vector<KVCacheBuffer>> k_cache_;
  std::vector<std::vector<KVCacheBuffer>> v_cache_;
  
  // KV cell metadata: [context_len] - shared across all layers/heads
  std::vector<KVCellMeta> cell_meta_;

  // Rearrange helper functions
  void rearrange_key(
      KVCacheBuffer& cache,
      int32_t src_cache_len,
      int32_t dst_cache_len);
  
  void rearrange_value(
      KVCacheBuffer& cache,
      int32_t src_cache_len,
      int32_t dst_cache_len);
  
  int32_t cur_ar_len_;  // Current AR length (for rearrange tracking)
};

} // namespace llama_qnn

