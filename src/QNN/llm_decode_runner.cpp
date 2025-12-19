#include "llm_decode_runner.h"
#include "llm_input_preparer.h"
#include "qnn_tensor_util.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <set>

namespace llama_qnn {

LLMDecodeRunner::LLMDecodeRunner(const LLMDecodeConfig& config)
    : config_(config),
      prefill_graph_(nullptr),
      kv_graph_(nullptr),
      context_len_(0),
      num_layers_(0),
      num_heads_(0),
      head_dim_(0),
      prefill_ar_len_(0),
      kv_ar_len_(0),
      prefill_cache_len_(0),
      kv_cache_len_(0),
      layers_per_shard_(0) {
}

LLMDecodeRunner::~LLMDecodeRunner() = default;

bool LLMDecodeRunner::initialize() {
  // Track model load time
  stats_.model_load_start_ms = time_in_ms();
  
  // 1. Load QNN backend
  loader_.reset(new QnnLoader());
  
  // Set QNN log level (1=ERROR, 2=WARN, 3=INFO, 4=VERBOSE, 5=DEBUG)
  loader_->set_log_level(config_.log_level);
  
  if (!loader_->load(config_.backend_so, config_.system_so)) {
    error_msg_ = "Failed to load QNN backend";
    return false;
  }
  
  if (!loader_->get_interface_provider()) {
    error_msg_ = "Failed to get QNN interface provider";
    return false;
  }
  
  if (!loader_->create_backend_and_device()) {
    error_msg_ = "Failed to create QNN backend and device";
    return false;
  }
  
  // Enable HTP Performance Mode (Burst)
  if (!loader_->enable_htp_performance_mode()) {
    if (config_.log_level >= 2) {
      std::cerr << "[Init] Warning: Failed to enable HTP performance mode\n";
    }
  }

  if (config_.log_level >= 1) {
    std::cout << "[Init] QNN backend loaded\n";
    if (config_.use_multi_context) {
      std::cout << "[Init] Multi-context mode enabled (" << config_.num_shards << " shards)\n";
    }
  }
  
  // 1.5. Load model parameters if provided
  if (!config_.params_path.empty()) {
    if (!parse_model_params(config_.params_path, model_params_)) {
      error_msg_ = "Failed to parse params.json: " + config_.params_path;
      return false;
    }
    
    if (config_.log_level >= 1) {
      std::cout << "[ModelParams] Loaded from: " << config_.params_path << "\n";
      std::cout << "  dim=" << model_params_.dim 
                << ", n_layers=" << model_params_.n_layers
                << ", n_heads=" << model_params_.n_heads
                << ", n_kv_heads=" << model_params_.n_kv_heads << "\n";
      std::cout << "  head_dim=" << model_params_.head_dim
                << ", vocab_size=" << model_params_.vocab_size << "\n";
    }
    
    // Compute layers per shard for multi-context
    if (config_.use_multi_context && config_.num_shards > 0) {
      layers_per_shard_ = model_params_.n_layers / config_.num_shards;
      if (config_.log_level >= 1) {
        std::cout << "  layers_per_shard=" << layers_per_shard_ 
                  << " (" << model_params_.n_layers << " / " << config_.num_shards << ")\n";
      }
      if (model_params_.n_layers % config_.num_shards != 0) {
        std::cerr << "[Warning] n_layers (" << model_params_.n_layers 
                  << ") is not evenly divisible by num_shards (" << config_.num_shards << ")\n";
      }
    }
  } else if (config_.log_level >= 1) {
    std::cout << "[ModelParams] No params.json provided, will extract from graphs\n";
  }
  
  // 2-5. Choose single vs multi-context initialization path
  if (config_.use_multi_context) {
    // Multi-context mode (sharding)
    if (!load_multi_context_graphs()) return false;
    if (!extract_multi_context_metadata()) return false;
    if (!setup_multi_context_kv_cache()) return false;
    if (!setup_multi_context_io_allocators()) return false;
    if (!allocate_shared_buffers()) return false;
  } else {
    // Single-context mode
    if (!load_graphs()) return false;
    if (!extract_metadata()) return false;
    if (!setup_kv_cache()) return false;
    if (!setup_io_allocators()) return false;
  }
  
  // 6. Load tokenizer
  tokenizer_.reset(new LlamaTokenizer());
  if (!tokenizer_->init(config_.tokenizer_path.c_str())) {
    error_msg_ = "Failed to load tokenizer";
    return false;
  }
  
  // Track model load end time
  stats_.model_load_end_ms = time_in_ms();
  
  if (config_.log_level >= 1) {
    std::cout << "[Init] All components initialized successfully\n";
    double load_time_s = (stats_.model_load_end_ms - stats_.model_load_start_ms) / 1000.0;
    std::cout << "[Init] Model load time: " << load_time_s << " seconds\n";
  }
  
  return true;
}

bool LLMDecodeRunner::load_graphs() {
  std::string json_path = config_.ctx_dir + "/forward_0_json.json";
  
  if (!parse_qnn_json(json_path, graphs_)) {
    error_msg_ = "Failed to parse QNN JSON: " + json_path;
    return false;
  }
  
  if (graphs_.find("prefill_forward") == graphs_.end() ||
      graphs_.find("kv_forward") == graphs_.end()) {
    error_msg_ = "Required graphs not found (prefill_forward, kv_forward)";
    return false;
  }
  
  prefill_graph_ = &graphs_["prefill_forward"];
  kv_graph_ = &graphs_["kv_forward"];
  
  if (config_.log_level >= 1) {
    std::cout << "[Graphs] Loaded prefill_forward and kv_forward\n";
  }
  
  // Load context binary
  std::string ctx_bin = config_.ctx_dir + "/forward_0.bin";
  
  // Read binary file
  std::ifstream ifs(ctx_bin, std::ios::binary | std::ios::ate);
  if (!ifs) {
    error_msg_ = "Failed to open context binary: " + ctx_bin;
    return false;
  }
  
  size_t size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (!ifs.read(buffer.data(), size)) {
    error_msg_ = "Failed to read context binary: " + ctx_bin;
    return false;
  }
  ifs.close();
  
  if (!loader_->create_context_from_binary(buffer.data(), size)) {
    error_msg_ = "Failed to create context from binary: " + ctx_bin;
    return false;
  }
  
  // Retrieve graphs
  if (!loader_->retrieve_graph(0, "prefill_forward") ||
      !loader_->retrieve_graph(0, "kv_forward")) {
    error_msg_ = "Failed to retrieve graphs";
    return false;
  }
  
  return true;
}

bool LLMDecodeRunner::extract_metadata() {
  // 1. Extract dimensions from graph (always needed from attention mask)
  context_len_ = 0;
  prefill_ar_len_ = 0;
  kv_ar_len_ = 0;
  
  // Find attention mask to get context_len and ar_len from prefill graph
  for (const auto& t : prefill_graph_->inputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    
    if (name_lower.find("atten_mask") != std::string::npos && t.dims.size() >= 2) {
      prefill_ar_len_ = t.dims[t.dims.size() - 2];
      context_len_ = t.dims[t.dims.size() - 1];
      break;
    }
  }
  
  // Extract kv_ar_len from kv graph
  for (const auto& t : kv_graph_->inputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    
    if (name_lower.find("atten_mask") != std::string::npos && t.dims.size() >= 2) {
      kv_ar_len_ = t.dims[t.dims.size() - 2];
      if (context_len_ == 0) {
        context_len_ = t.dims[t.dims.size() - 1];
      }
      break;
    }
  }
  
  // 2. Use params.json if available, otherwise infer from graph
  if (model_params_.is_valid()) {
    // Use values from params.json
    num_layers_ = model_params_.n_layers;
    num_heads_ = model_params_.n_kv_heads;
    head_dim_ = model_params_.head_dim;
  } else {
    // Fallback: extract from graph tensors
    head_dim_ = 0;
    
    // Extract head_dim from KV cache tensor
    for (const auto& t : prefill_graph_->inputs) {
      if (t.name.find("_args_") != std::string::npos && t.dims.size() == 3) {
        head_dim_ = t.dims[2];
        if (head_dim_ > 0) break;
      }
    }
    
    // Count KV cache tensors to infer num_layers and num_heads
    std::set<std::string> kv_cache_names;
    for (const auto& t : prefill_graph_->inputs) {
      if (t.name.find("_args_") != std::string::npos && t.dims.size() == 3) {
        kv_cache_names.insert(t.name);
      }
    }
    
    int total_kv_tensors = kv_cache_names.size();
    num_layers_ = 16;  // Default: llama 1B
    num_heads_ = total_kv_tensors / (2 * num_layers_);
  }
  
  // Cache lengths (ExecutorchReader style)
  prefill_cache_len_ = context_len_ - prefill_ar_len_;  // 512 - 32 = 480
  kv_cache_len_ = context_len_ - kv_ar_len_;            // 512 - 1 = 511
  
  if (config_.log_level >= 1) {
    std::cout << "[Metadata] context_len=" << context_len_
              << ", prefill_ar=" << prefill_ar_len_
              << ", kv_ar=" << kv_ar_len_ << "\n";
    std::cout << "[Metadata] num_layers=" << num_layers_
              << ", num_heads=" << num_heads_
              << ", head_dim=" << head_dim_ << "\n";
    std::cout << "[Metadata] prefill_cache_len=" << prefill_cache_len_
              << ", kv_cache_len=" << kv_cache_len_ << "\n";
    if (model_params_.is_valid()) {
      std::cout << "[Metadata] Source: params.json ✓\n";
    } else {
      std::cout << "[Metadata] Source: inferred from graph tensors\n";
    }
  }
  
  if (context_len_ == 0 || num_heads_ == 0 || head_dim_ == 0) {
    error_msg_ = "Failed to extract valid metadata";
    return false;
  }
  
  return true;
}

bool LLMDecodeRunner::setup_kv_cache() {
  LLMKVCacheManager::Metadata kv_meta{
      context_len_,
      head_dim_,
      prefill_ar_len_,
      kv_cache_len_,
      num_heads_,
      num_layers_
  };
  
  kv_manager_.reset(new LLMKVCacheManager(kv_meta));
  if (!kv_manager_->allocate()) {
    error_msg_ = "Failed to allocate KV cache memory";
    return false;
  }
  
  if (config_.log_level >= 1) {
    std::cout << "[KV Cache] Allocated "
              << (kv_manager_->total_cache_size() / 1024.0 / 1024.0)
              << " MiB\n";
  }
  
  // Build KV cache mappings
  prefill_kv_mapping_ = LLMKVCacheMapper::build_mapping(
      *prefill_graph_, num_heads_, head_dim_);
  kv_kv_mapping_ = LLMKVCacheMapper::build_mapping(
      *kv_graph_, num_heads_, head_dim_);
  
  prefill_kv_override_ = LLMKVCacheMapper::create_buffer_override(
      prefill_kv_mapping_, *kv_manager_);
  kv_kv_override_ = LLMKVCacheMapper::create_buffer_override(
      kv_kv_mapping_, *kv_manager_);
  
  if (config_.log_level >= 1) {
    std::cout << "[KV Binding] Prefill: " << prefill_kv_mapping_.size()
              << " tensors, Decode: " << kv_kv_mapping_.size() << " tensors\n";
  }
  
  return true;
}

bool LLMDecodeRunner::setup_io_allocators() {
  // 1. Allocate I/O buffers
  prefill_alloc_.reset(new QNNIOAllocator());
  prefill_alloc_->build_from_qnnjson(*prefill_graph_);
  auto prefill_bytes = prefill_alloc_->allocate(64);
  
  kv_alloc_.reset(new QNNIOAllocator());
  kv_alloc_->build_from_qnnjson(*kv_graph_);
  auto kv_bytes = kv_alloc_->allocate(64);
  
  // 2. Pre-build QNN tensor holders (one-time setup)
  prefill_input_holders_.clear();
  prefill_output_holders_.clear();
  kv_input_holders_.clear();
  kv_output_holders_.clear();
  
  // Prefill input holders
  for (const auto& t : prefill_graph_->inputs) {
    auto h = std::make_unique<QnnTensorHolder>();
    // Initialize with nullptr, will update pointer before execution
    h->init_from_json(t, nullptr, t.nbytes, true);
    prefill_input_holders_.push_back(std::move(h));
  }
  
  // Prefill output holders
  for (const auto& t : prefill_graph_->outputs) {
    auto h = std::make_unique<QnnTensorHolder>();
    h->init_from_json(t, nullptr, t.nbytes, false);
    prefill_output_holders_.push_back(std::move(h));
  }
  
  // KV input holders
  for (const auto& t : kv_graph_->inputs) {
    auto h = std::make_unique<QnnTensorHolder>();
    h->init_from_json(t, nullptr, t.nbytes, true);
    kv_input_holders_.push_back(std::move(h));
  }
  
  // KV output holders
  for (const auto& t : kv_graph_->outputs) {
    auto h = std::make_unique<QnnTensorHolder>();
    h->init_from_json(t, nullptr, t.nbytes, false);
    kv_output_holders_.push_back(std::move(h));
  }
  
  if (config_.log_level >= 1) {
    std::cout << "[I/O] Prefill: " << (prefill_bytes / 1024.0)
              << " KiB, Decode: " << (kv_bytes / 1024.0) << " KiB\n";
    std::cout << "[I/O] Pre-built tensors - Prefill: " 
              << prefill_input_holders_.size() << " in, "
              << prefill_output_holders_.size() << " out / Decode: "
              << kv_input_holders_.size() << " in, "
              << kv_output_holders_.size() << " out\n";
  }
  
  return true;
}

// bool LLMDecodeRunner::generate(const std::vector<int32_t>& prompt_tokens, // will be deprecated
//                                std::vector<int32_t>& generated_tokens) {
//   // Start inference timing
//   stats_.inference_start_ms = time_in_ms();
  
//   // 1. Tokenize prompt (no special tokens, no chat template - same as qnn_decode_main)
//   // auto tokens = tokenizer_->encode(prompt, false, false); // [spagetti] 토크나이저가 느릴 가능성은? - 별로 안중요
//   // if (tokens.empty()) {
//   //   error_msg_ = "Failed to tokenize prompt";
//   //   return false;
//   // }

//   std::vector<int32_t> tokens = prompt_tokens;
  
//   stats_.num_prompt_tokens = tokens.size();
  
//   // if (config_.log_level >= 1) {
//   //   std::cout << "\n[Generate] Prompt: \"" << prompt << "\"\n";
//   //   std::cout << "[Generate] Tokens: " << tokens.size() << "\n";
//   // }
  
//   // 3. Run prefill (choose single vs multi-context)
//   int32_t next_token = 0;
//   int32_t n_update = 0;
//   if (config_.use_multi_context) {
//     if (!run_multi_context_prefill(tokens, next_token, n_update)) {
//       return false;
//     }
//   } else {
//     if (!run_prefill(tokens, next_token, n_update)) {
//       return false;
//     }
//   }
  
//   // Mark prefill end (TTFT)
//   stats_.prompt_eval_end_ms = time_in_ms();
//   stats_.first_token_ms = stats_.prompt_eval_end_ms;
  
//   // 4. Decode first token
//   std::string decoded = tokenizer_->decode({next_token});
//   // output_text = decoded;
  
//   if (config_.log_level >= 1) {
//     std::cout << "[Prefill] Next token: " << next_token
//               << " → \"" << decoded << "\"\n";
//     double ttft_s = (stats_.first_token_ms - stats_.inference_start_ms) / 1000.0;
//     std::cout << "[Prefill] TTFT: " << ttft_s << " seconds\n";
//   }
  
//   tokens.push_back(next_token);
//   stats_.num_generated_tokens = 1;
  
//   // 4. Rearrange cache for decode (single-context only, multi-context does it internally)
//   if (!config_.use_multi_context) {
//     if (config_.log_level >= 1) {
//       std::cout << "\n[Rearrange] Expanding KV cache: "
//                 << prefill_cache_len_ << " → " << kv_cache_len_ << "\n";
//     }
//     kv_manager_->rearrange_cache(prefill_ar_len_, kv_ar_len_);
//   }
  
//   // 5. Decode loop
//   if (config_.log_level >= 1) {
//     std::cout << "\n[Decode] Generating up to " << config_.max_gen_tokens
//               << " tokens...\n";
//     std::cout << "[Output] " << decoded;
//     std::cout.flush();
//   }
  
//   int32_t initial_tokens = n_update;
  
//   for (int gen_idx = 0; gen_idx < config_.max_gen_tokens - 1; ++gen_idx) {
//     int32_t n_past = initial_tokens + gen_idx;
//     int32_t token_out = 0;
    
//     // Run decode step (choose single vs multi-context)
//     if (config_.use_multi_context) {
//       if (!run_multi_context_decode_step(next_token, n_past, token_out)) {
//         return false;
//       }
//     } else {
//       if (!run_decode_step(next_token, n_past, token_out)) {
//         return false;
//       }
//     }
    
//     // Check EOS
//     if (token_out == 128001 || token_out == 128009) {
//       if (config_.log_level >= 1) {
//         std::cout << "\n[Decode] EOS token detected\n";
//       }
//       break;
//     }
    
//     // Decode and append
//     decoded = tokenizer_->decode({token_out});
//     // output_text += decoded;
    
//     if (config_.log_level >= 1) {
//       std::cout << decoded;
//       std::cout.flush();
//     }
    
//     next_token = token_out;
//     tokens.push_back(token_out);
//     stats_.num_generated_tokens++;
//   }
  
//   // Mark inference end
//   stats_.inference_end_ms = time_in_ms();
  
//   if (config_.log_level >= 1) {
//     std::cout << "\n\n[Generate] Complete. Total tokens: " << tokens.size() << "\n";
//   }
  
//   // Print performance report
//   if (config_.log_level >= 1) {
//     stats_.print_report();
//   }
  
//   return true;
// }

bool LLMDecodeRunner::run_prefill(const std::vector<int32_t>& tokens,
                                   int32_t& next_token,
                                   int32_t& n_update,
                                   llama_context * llama_ctx) {
  // Prepare inputs
  auto get_prefill_buffer = [&](const std::string& name) -> void* {
    auto it = prefill_kv_override_.find(name);
    if (it != prefill_kv_override_.end()) return it->second;
    
    auto& bindings = prefill_alloc_->bindings();
    auto bit = bindings.find(name);
    return (bit != bindings.end()) ? bit->second : nullptr;
  };
  
  if (!InputPreparer::auto_fill_inputs(*prefill_graph_, get_prefill_buffer, tokens, true)) {
    error_msg_ = "Failed to prepare prefill inputs";
    return false;
  }
  
  // Update pre-built tensors with current buffer pointers (zero allocation)
  std::vector<Qnn_Tensor_t> inputs, outputs;
  
  for (size_t i = 0; i < prefill_graph_->inputs.size() && i < prefill_input_holders_.size(); ++i) {
    const auto& t = prefill_graph_->inputs[i];
    void* buf = get_prefill_buffer(t.name);
    if (!buf) {
      if (config_.log_level >= 2) {
        std::cerr << "[Prefill] Warning: No buffer for input " << t.name << "\n";
      }
      continue;
    }
    
    // Update buffer pointer only (no allocation)
    prefill_input_holders_[i]->update_buffer(buf, t.nbytes);
    inputs.push_back(prefill_input_holders_[i]->tensor());
  }
  
  for (size_t i = 0; i < prefill_graph_->outputs.size() && i < prefill_output_holders_.size(); ++i) {
    const auto& t = prefill_graph_->outputs[i];
    auto& bindings = prefill_alloc_->bindings();
    auto it = bindings.find(t.name);
    if (it == bindings.end()) {
      if (config_.log_level >= 2) {
        std::cerr << "[Prefill] Warning: No buffer for output " << t.name << "\n";
      }
      continue;
    }
    
    // Update buffer pointer only (no allocation)
    prefill_output_holders_[i]->update_buffer(it->second, t.nbytes);
    outputs.push_back(prefill_output_holders_[i]->tensor());
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Prefill] Prepared " << inputs.size() << " inputs, "
              << outputs.size() << " outputs\n";
  }
  
  // Execute
  if (!loader_->execute_graph(0, "prefill_forward", inputs, outputs)) {
    error_msg_ = "Prefill execution failed";
    return false;
  }
  
  // Extract logits and decode token
  const QnnJsonTensorDesc* logits_desc = nullptr;
  for (const auto& t : prefill_graph_->outputs) {
    if (t.name.find("squeeze") != std::string::npos ||
        t.name.find("logit") != std::string::npos) {
      logits_desc = &t;
      break;
    }
  }
  
  if (!logits_desc) {
    error_msg_ = "Logits output not found";
    return false;
  }
  
  auto& bindings = prefill_alloc_->bindings();
  auto it = bindings.find(logits_desc->name);
  if (it == bindings.end()) {
    error_msg_ = "Logits buffer not found";
    return false;
  }
  
  const void * logits_q = it->second;

  // Dequantize full logits tensor
  std::vector<float> logits_f32;
  if (!OutputProcessor::dequantize_logits(logits_q, *logits_desc, logits_f32)) {
    error_msg_ = "Failed to dequantize prefill logits";
    return false;
  }

  // Determine vocab size and last-token row offset
  int32_t vocab_size = 1;
  if (!logits_desc->dims.empty()) {
    vocab_size = (int32_t) logits_desc->dims.back();
  }

  int32_t last_token_index = (int32_t) tokens.size() - 1;
  if (last_token_index < 0) {
    error_msg_ = "No tokens for prefill";
    return false;
  }

  size_t last_token_offset = (size_t) last_token_index * (size_t) vocab_size;
  if (last_token_offset + (size_t) vocab_size > logits_f32.size()) {
    error_msg_ = "Prefill logits buffer too small";
    return false;
  }

  // Greedy argmax on last token logits (backward compatibility)
  const float * last_row = logits_f32.data() + last_token_offset;
  float max_val = last_row[0];
  next_token = 0;
  for (int32_t i = 1; i < vocab_size; ++i) {
    if (last_row[i] > max_val) {
      max_val = last_row[i];
      next_token = i;
    }
  }

  // Inject logits into llama_context for external sampling if provided
  if (llama_ctx != nullptr) {
    llama_set_logits_external(llama_ctx, last_row, 1);
  }
  
  // Update KV cache from prefill outputs
  n_update = 1 + ((tokens.size() - 1) % prefill_ar_len_);
  int32_t n_past = 0;
  
  int v_idx = 0, k_idx = 0;
  for (const auto& t : prefill_graph_->outputs) {
    std::string n = t.name;
    
    bool is_v = (n.find("view_copy") != std::string::npos &&
                 t.dims.size() == 3 && t.dims[1] == prefill_ar_len_ && t.dims[2] == head_dim_);
    bool is_k = (n.find("permute_copy") != std::string::npos &&
                 t.dims.size() == 3 && t.dims[1] == head_dim_ && t.dims[2] == prefill_ar_len_);
    
    if (!is_v && !is_k) continue;
    
    auto bit = bindings.find(t.name);
    if (bit == bindings.end()) continue;
    
    if (is_v) {
      int layer = v_idx / num_heads_;
      int head = v_idx % num_heads_;
      v_idx++;
      
      if (layer >= num_layers_ || head >= num_heads_) continue;
      
      const auto& v_buf = kv_manager_->get_v_cache(layer, head);
      uint8_t* src = reinterpret_cast<uint8_t*>(bit->second);
      uint8_t* dst = reinterpret_cast<uint8_t*>(v_buf.input_buffer) + n_past * head_dim_;
      std::memcpy(dst, src, n_update * head_dim_);
      
    } else if (is_k) {
      int layer = k_idx / num_heads_;
      int head = k_idx % num_heads_;
      k_idx++;
      
      if (layer >= num_layers_ || head >= num_heads_) continue;
      
      const auto& k_buf = kv_manager_->get_k_cache(layer, head);
      uint8_t* src = reinterpret_cast<uint8_t*>(bit->second);
      uint8_t* dst = reinterpret_cast<uint8_t*>(k_buf.input_buffer) + n_past;
      
      for (int32_t dim = 0; dim < head_dim_; ++dim) {
        std::memcpy(dst, src, n_update);
        src += prefill_ar_len_;
        dst += prefill_cache_len_;
      }
    }
  }
  
  return true;
}

bool LLMDecodeRunner::run_decode_step(int32_t token_in,
                                       int32_t n_past,
                                       int32_t& token_out,
                                       llama_context * llama_ctx) {
  // Prepare inputs
  auto get_kv_buffer = [&](const std::string& name) -> void* {
    auto it = kv_kv_override_.find(name);
    if (it != kv_kv_override_.end()) return it->second;
    
    auto& bindings = kv_alloc_->bindings();
    auto bit = bindings.find(name);
    return (bit != bindings.end()) ? bit->second : nullptr;
  };
  
  // Fill inputs
  for (const auto& t : kv_graph_->inputs) {
    std::string n = t.name;
    for (auto& c : n) c = (char)tolower(c);
    
    void* buf = get_kv_buffer(t.name);
    if (!buf) continue;
    
    // Token
    if (n.find("token") != std::string::npos && n.find("input") != std::string::npos) {
      std::memcpy(buf, &token_in, sizeof(int32_t));
    }
    // Position
    else if (n.find("pos") != std::string::npos && t.data_type.find("INT_32") != std::string::npos) {
      std::memcpy(buf, &n_past, sizeof(int32_t));
    }
    // Attention mask
    else if (n.find("atten_mask") != std::string::npos) {
      uint16_t* mask = reinterpret_cast<uint16_t*>(buf);
      std::memset(mask, 0, t.nbytes);
      
      // Attend to past tokens [0..n_past-1]
      for (int32_t i = 0; i < n_past; ++i) {
        mask[i] = 65535;
      }
      // Attend to current token (last position)
      mask[context_len_ - 1] = 65535;
    }
  }
  
  // Update pre-built tensors with current buffer pointers (zero allocation)
  std::vector<Qnn_Tensor_t> inputs, outputs;
  
  for (size_t i = 0; i < kv_graph_->inputs.size() && i < kv_input_holders_.size(); ++i) {
    const auto& t = kv_graph_->inputs[i];
    void* buf = get_kv_buffer(t.name);
    if (!buf) {
      if (config_.log_level >= 2) {
        std::cerr << "[Decode] Warning: No buffer for input " << t.name << "\n";
      }
      continue;
    }
    
    // Update buffer pointer only (no allocation)
    kv_input_holders_[i]->update_buffer(buf, t.nbytes);
    inputs.push_back(kv_input_holders_[i]->tensor());
  }
  
  for (size_t i = 0; i < kv_graph_->outputs.size() && i < kv_output_holders_.size(); ++i) {
    const auto& t = kv_graph_->outputs[i];
    auto& bindings = kv_alloc_->bindings();
    auto it = bindings.find(t.name);
    if (it == bindings.end()) {
      if (config_.log_level >= 2) {
        std::cerr << "[Decode] Warning: No buffer for output " << t.name << "\n";
      }
      continue;
    }
    
    // Update buffer pointer only (no allocation)
    kv_output_holders_[i]->update_buffer(it->second, t.nbytes);
    outputs.push_back(kv_output_holders_[i]->tensor());
  }
  
  // Execute
  if (!loader_->execute_graph(0, "kv_forward", inputs, outputs)) {
    error_msg_ = "Decode execution failed";
    return false;
  }
  
  // Extract logits
  const QnnJsonTensorDesc* logits_desc = nullptr;
  for (const auto& t : kv_graph_->outputs) {
    if (t.name.find("squeeze") != std::string::npos ||
        t.name.find("logit") != std::string::npos) {
      logits_desc = &t;
      break;
    }
  }
  
  if (!logits_desc) {
    error_msg_ = "Logits output not found";
    return false;
  }
  
  auto& bindings = kv_alloc_->bindings();
  auto it = bindings.find(logits_desc->name);
  if (it == bindings.end()) {
    error_msg_ = "Logits buffer not found";
    return false;
  }
  
  const void * logits_q = it->second;

  // Dequantize logits for this decode step
  std::vector<float> logits_f32;
  if (!OutputProcessor::dequantize_logits(logits_q, *logits_desc, logits_f32)) {
    error_msg_ = "Failed to dequantize decode logits";
    return false;
  }

  int32_t vocab_size = 1;
  if (!logits_desc->dims.empty()) {
    vocab_size = (int32_t) logits_desc->dims.back();
  }

  if ((size_t) vocab_size > logits_f32.size()) {
    error_msg_ = "Decode logits buffer too small";
    return false;
  }

  const float * row = logits_f32.data();

  // Greedy argmax for backward compatibility
  float max_val = row[0];
  token_out = 0;
  for (int32_t i = 1; i < vocab_size; ++i) {
    if (row[i] > max_val) {
      max_val = row[i];
      token_out = i;
    }
  }

  // Inject logits into llama_context for external sampling if provided
  if (llama_ctx != nullptr) {
    llama_set_logits_external(llama_ctx, row, 1);
  }
  
  // Update KV cache from decode outputs
  int v_idx = 0, k_idx = 0;
  for (const auto& t : kv_graph_->outputs) {
    std::string n = t.name;
    
    bool is_v = (n.find("view_copy") != std::string::npos &&
                 t.dims.size() == 3 && t.dims[1] == kv_ar_len_ && t.dims[2] == head_dim_);
    bool is_k = (n.find("permute_copy") != std::string::npos &&
                 t.dims.size() == 3 && t.dims[1] == head_dim_ && t.dims[2] == kv_ar_len_);
    
    if (!is_v && !is_k) continue;
    
    auto bit = bindings.find(t.name);
    if (bit == bindings.end()) continue;
    
    if (is_v) {
      int layer = v_idx / num_heads_;
      int head = v_idx % num_heads_;
      v_idx++;
      
      if (layer >= num_layers_ || head >= num_heads_) continue;
      
      const auto& v_buf = kv_manager_->get_v_cache(layer, head);
      uint8_t* src = reinterpret_cast<uint8_t*>(bit->second);
      uint8_t* dst = reinterpret_cast<uint8_t*>(v_buf.input_buffer) + n_past * head_dim_;
      std::memcpy(dst, src, kv_ar_len_ * head_dim_);
      
    } else if (is_k) {
      int layer = k_idx / num_heads_;
      int head = k_idx % num_heads_;
      k_idx++;
      
      if (layer >= num_layers_ || head >= num_heads_) continue;
      
      const auto& k_buf = kv_manager_->get_k_cache(layer, head);
      uint8_t* src = reinterpret_cast<uint8_t*>(bit->second);
      uint8_t* dst = reinterpret_cast<uint8_t*>(k_buf.input_buffer) + n_past;
      
      for (int32_t dim = 0; dim < head_dim_; ++dim) {
        std::memcpy(dst, src, kv_ar_len_);
        src += kv_ar_len_;
        dst += kv_cache_len_;
      }
    }
  }
  
  return true;
}

} // namespace llama_qnn
