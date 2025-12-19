#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llama_qnn {

class LlamaTokenizer {
public:
  LlamaTokenizer();
  ~LlamaTokenizer();

  bool init(const char* gguf_path);
  void shutdown();

  // Llama 3.2 템플릿을 적용한 후 encode 호출 권장
  std::vector<int32_t> encode(const std::string& text, bool add_special = true, bool parse_special = true);
  std::string decode(const std::vector<int32_t>& tokens, bool special = true);

private:
  void* model_;
};

// Llama 3.2 포맷팅 헬퍼
std::string format_llama32_prompt(const std::string& user, const std::string& system);

} // namespace llama_qnn


