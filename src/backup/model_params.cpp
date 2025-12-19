#include "model_params.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

namespace llama_qnn {

// Simple JSON parser for params.json (lightweight, no external dependencies)
static bool parse_json_field(const std::string& json, const std::string& key, int32_t& out_value) {
  std::string search = "\"" + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) return false;
  
  pos += search.length();
  // Skip whitespace
  while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  
  // Parse number
  size_t end_pos = pos;
  bool is_negative = false;
  if (json[end_pos] == '-') {
    is_negative = true;
    ++end_pos;
  }
  while (end_pos < json.length() && json[end_pos] >= '0' && json[end_pos] <= '9') {
    ++end_pos;
  }
  
  if (end_pos > pos) {
    std::string num_str = json.substr(pos, end_pos - pos);
    try {
      out_value = std::stoi(num_str);
      return true;
    } catch (...) {
      return false;
    }
  }
  return false;
}

static bool parse_json_field(const std::string& json, const std::string& key, float& out_value) {
  std::string search = "\"" + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) return false;
  
  pos += search.length();
  // Skip whitespace
  while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  
  // Parse number (int or float)
  size_t end_pos = pos;
  if (json[end_pos] == '-') ++end_pos;
  while (end_pos < json.length() && 
         ((json[end_pos] >= '0' && json[end_pos] <= '9') || 
          json[end_pos] == '.' || json[end_pos] == 'e' || json[end_pos] == 'E' ||
          json[end_pos] == '-' || json[end_pos] == '+')) {
    ++end_pos;
  }
  
  if (end_pos > pos) {
    std::string num_str = json.substr(pos, end_pos - pos);
    try {
      out_value = std::stof(num_str);
      return true;
    } catch (...) {
      return false;
    }
  }
  return false;
}

static bool parse_json_field(const std::string& json, const std::string& key, bool& out_value) {
  std::string search = "\"" + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) return false;
  
  pos += search.length();
  // Skip whitespace
  while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  
  if (json.substr(pos, 4) == "true") {
    out_value = true;
    return true;
  } else if (json.substr(pos, 5) == "false") {
    out_value = false;
    return true;
  }
  return false;
}

bool parse_model_params(const std::string& path, ModelParams& params) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[ModelParams] Failed to open: " << path << "\n";
    return false;
  }
  
  // Read entire file into string
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json = buffer.str();
  
  // Parse fields (optional, so don't fail if missing)
  parse_json_field(json, "dim", params.dim);
  parse_json_field(json, "n_layers", params.n_layers);
  parse_json_field(json, "n_heads", params.n_heads);
  parse_json_field(json, "n_kv_heads", params.n_kv_heads);
  parse_json_field(json, "vocab_size", params.vocab_size);
  parse_json_field(json, "ffn_dim_multiplier", params.ffn_dim_multiplier);
  parse_json_field(json, "multiple_of", params.multiple_of);
  parse_json_field(json, "norm_eps", params.norm_eps);
  parse_json_field(json, "rope_theta", params.rope_theta);
  parse_json_field(json, "use_scaled_rope", params.use_scaled_rope);
  
  // Compute derived values
  params.compute_derived();
  
  if (!params.is_valid()) {
    std::cerr << "[ModelParams] Invalid parameters in: " << path << "\n";
    return false;
  }
  
  return true;
}

} // namespace llama_qnn
