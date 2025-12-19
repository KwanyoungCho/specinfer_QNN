#pragma once

#include <string>
#include <cstdint>

namespace llama_qnn {

/**
 * @brief Model parameters from params.json
 */
struct ModelParams {
  int32_t dim = 0;                    // Hidden dimension (e.g., 2048)
  int32_t n_layers = 0;               // Total number of layers (e.g., 16)
  int32_t n_heads = 0;                // Number of query heads (e.g., 32)
  int32_t n_kv_heads = 0;             // Number of key-value heads (e.g., 8)
  int32_t vocab_size = 0;             // Vocabulary size (e.g., 128256)
  float ffn_dim_multiplier = 1.0f;    // FFN dimension multiplier (e.g., 1.5)
  int32_t multiple_of = 256;          // FFN rounding multiple
  float norm_eps = 1e-5f;             // Normalization epsilon
  float rope_theta = 10000.0f;        // RoPE theta base
  bool use_scaled_rope = false;       // Use scaled RoPE
  
  // Derived values (computed from above)
  int32_t head_dim = 0;               // dim / n_heads
  int32_t hidden_dim = 0;             // Same as dim
  int32_t layers_per_shard = 0;       // n_layers / num_shards (for multi-context)
  
  /**
   * @brief Compute derived values
   */
  void compute_derived() {
    if (n_heads > 0) {
      head_dim = dim / n_heads;
    }
    hidden_dim = dim;
  }
  
  /**
   * @brief Validate parameters
   */
  bool is_valid() const {
    return dim > 0 && n_layers > 0 && n_heads > 0 && n_kv_heads > 0 &&
           vocab_size > 0 && head_dim > 0;
  }
};

/**
 * @brief Parse params.json file
 * @param path Path to params.json
 * @param params Output parameter structure
 * @return true on success
 */
bool parse_model_params(const std::string& path, ModelParams& params);

} // namespace llama_qnn
