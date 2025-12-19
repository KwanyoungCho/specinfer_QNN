#include "tokenizer_llama.h"

#include <llama.h>
#include <mutex>

namespace llama_qnn {

static std::mutex g_tok_mu;

LlamaTokenizer::LlamaTokenizer() : model_(nullptr) {}

LlamaTokenizer::~LlamaTokenizer() { shutdown(); }

bool LlamaTokenizer::init(const char* gguf_path) {
  std::lock_guard<std::mutex> lk(g_tok_mu);
  if (model_) return true;
  
  // Disable llama.cpp logging
  llama_log_set([](enum ggml_log_level level, const char* text, void* user_data) {
    // Silent: do nothing
  }, nullptr);
  
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.use_mmap = false;
  mp.use_mlock = false;
  mp.vocab_only = true;
  model_ = llama_model_load_from_file(gguf_path, mp);
  return model_ != nullptr;
}

void LlamaTokenizer::shutdown() {
  std::lock_guard<std::mutex> lk(g_tok_mu);
  if (model_) {
    llama_model_free(reinterpret_cast<llama_model*>(model_));
    model_ = nullptr;
  }
  llama_backend_free();
}

std::vector<int32_t> LlamaTokenizer::encode(const std::string& text, bool add_special, bool parse_special) {
  std::lock_guard<std::mutex> lk(g_tok_mu);
  std::vector<int32_t> out;
  if (!model_) return out;
  const llama_vocab* vocab = llama_model_get_vocab(reinterpret_cast<llama_model*>(model_));
  // first pass to get required size
  int n = llama_tokenize(vocab, text.c_str(), (int)text.size(), nullptr, 0, add_special, parse_special);
  if (n < 0) n = -n; // required size
  std::vector<llama_token> tmp(n > 0 ? n : (int)text.size() + 16);
  n = llama_tokenize(vocab, text.c_str(), (int)text.size(), tmp.data(), (int)tmp.size(), add_special, parse_special);
  if (n < 0) n = 0;
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = (int32_t)tmp[i];
  return out;
}

std::string LlamaTokenizer::decode(const std::vector<int32_t>& tokens, bool special) {
  std::lock_guard<std::mutex> lk(g_tok_mu);
  std::string out;
  if (!model_) return out;
  const llama_vocab* vocab = llama_model_get_vocab(reinterpret_cast<llama_model*>(model_));
  if (tokens.empty()) return out;
  // try with a reasonable buffer, then retry if needed
  int32_t cap = (int32_t)(tokens.size() * 8 + 32);
  std::vector<char> buf(cap);
  int32_t n = llama_detokenize(vocab, reinterpret_cast<const llama_token*>(tokens.data()), (int32_t)tokens.size(), buf.data(), cap, /*remove_special=*/false, /*unparse_special=*/special);
  if (n < 0) {
    int32_t need = -n + 1;
    buf.resize(need);
    n = llama_detokenize(vocab, reinterpret_cast<const llama_token*>(tokens.data()), (int32_t)tokens.size(), buf.data(), need, /*remove_special=*/false, /*unparse_special=*/special);
    if (n < 0) n = 0;
  }
  out.assign(buf.data(), (size_t)std::max(0, n));
  return out;
}

std::string format_llama32_prompt(const std::string& user, const std::string& system) {
  std::string s;
  if (!system.empty()) {
    s += "<|start_header_id|>system<|end_header_id|>\n\n";
    s += system;
    s += "<|eot_id|>";
  }
  s += "<|start_header_id|>user<|end_header_id|>\n\n";
  s += user;
  s += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
  return s;
}

} // namespace llama_qnn


