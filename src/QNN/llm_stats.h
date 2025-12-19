/**
 * @file llm_stats.h
 * @brief Performance statistics tracking for LLM inference
 * Based on Executorch stats.h
 */

#pragma once

#include <cstdint>
#include <string>
#include <sstream>
#include <chrono>
#include <iostream>

namespace llama_qnn {

/**
 * @brief Get current time in milliseconds
 */
inline long time_in_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/**
 * @brief Performance statistics for LLM inference
 * 
 * Tracks timing for:
 * - Model loading
 * - Prefill (prompt evaluation)
 * - Decode (token generation)
 * - TTFT (Time To First Token)
 * - TPS (Tokens Per Second)
 */
struct LLMStats {
  // Scaling factor for timestamps (milliseconds)
  static constexpr long SCALING_FACTOR = 1000;
  
  // Timestamps
  long model_load_start_ms = 0;
  long model_load_end_ms = 0;
  long inference_start_ms = 0;
  long prompt_eval_end_ms = 0;  // After prefill
  long first_token_ms = 0;      // TTFT
  long inference_end_ms = 0;
  
  // Token counts
  int64_t num_prompt_tokens = 0;
  int64_t num_generated_tokens = 0;
  
  void reset() {
    model_load_start_ms = 0;
    model_load_end_ms = 0;
    inference_start_ms = 0;
    prompt_eval_end_ms = 0;
    first_token_ms = 0;
    inference_end_ms = 0;
    num_prompt_tokens = 0;
    num_generated_tokens = 0;
  }
  
  /**
   * @brief Print performance report
   */
  void print_report() const {
    std::cout << "\n========== Performance Report ==========\n";
    std::cout << "  Prompt Tokens: " << num_prompt_tokens << "\n";
    std::cout << "  Generated Tokens: " << num_generated_tokens << "\n";
    std::cout << "\n";
    
    // Model load time
    double model_load_time_s = (double)(model_load_end_ms - model_load_start_ms) / SCALING_FACTOR;
    std::cout << "  Model Load Time: " << model_load_time_s << " seconds\n";
    std::cout << "\n";
    
    // Time to first token (TTFT) - prefill time
    double ttft_s = (double)(first_token_ms - inference_start_ms) / SCALING_FACTOR;
    std::cout << "  TTFT (Prefill): " << ttft_s << " seconds";
    if (num_prompt_tokens > 0) {
      double prefill_tps = num_prompt_tokens / ttft_s;
      std::cout << " (" << prefill_tps << " tokens/second)";
    }
    std::cout << "\n";
    
    // Time between tokens (TBT) - decode time
    double decode_time_s = (double)(inference_end_ms - prompt_eval_end_ms) / SCALING_FACTOR;
    std::cout << "  TBT (Decode): " << decode_time_s << " seconds";
    if (num_generated_tokens > 0) {
      double decode_tps = num_generated_tokens / decode_time_s;
      std::cout << " (" << decode_tps << " tokens/second)";
      
      // Average time per token
      double avg_tbt_ms = (decode_time_s * 1000.0) / num_generated_tokens;
      std::cout << " [" << avg_tbt_ms << " ms/token]";
    }
    std::cout << "\n";
    
    // Total inference time
    double total_time_s = (double)(inference_end_ms - inference_start_ms) / SCALING_FACTOR;
    std::cout << "  Total Inference: " << total_time_s << " seconds";
    if (num_prompt_tokens + num_generated_tokens > 0) {
      double total_tps = (num_prompt_tokens + num_generated_tokens) / total_time_s;
      std::cout << " (" << total_tps << " tokens/second)";
    }
    std::cout << "\n";
    std::cout << "========================================\n\n";
  }
  
  /**
   * @brief Convert stats to JSON string
   */
  std::string to_json() const {
    std::stringstream ss;
    ss << "{"
       << "\"prompt_tokens\":" << num_prompt_tokens << ","
       << "\"generated_tokens\":" << num_generated_tokens << ","
       << "\"model_load_start_ms\":" << model_load_start_ms << ","
       << "\"model_load_end_ms\":" << model_load_end_ms << ","
       << "\"inference_start_ms\":" << inference_start_ms << ","
       << "\"prompt_eval_end_ms\":" << prompt_eval_end_ms << ","
       << "\"first_token_ms\":" << first_token_ms << ","
       << "\"inference_end_ms\":" << inference_end_ms << ","
       << "\"SCALING_FACTOR\":" << SCALING_FACTOR
       << "}";
    return ss.str();
  }
};

} // namespace llama_qnn
