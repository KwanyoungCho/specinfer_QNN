/**
 * @file llm_decode_runner_multi_context.cpp
 * @brief Multi-context (sharding) implementation for LLMDecodeRunner
 * 
 * Shard structure (16 layers, 2 layers per shard, 8 shards):
 * - Shard 0 (forward_0.bin):
 *   Inputs (35):  32 KV cache in + 1 token + 1 position + 1 attention_mask
 *   Outputs (35): 32 KV cache out + 1 hidden_state + 2 ROPE (cos, sin)
 * 
 * - Shard 1-7 (forward_1.bin ~ forward_7.bin):
 *   Inputs (36):  32 KV cache in + 1 hidden_state + 2 ROPE + 1 attention_mask
 *   Outputs (33): 32 KV cache out + 1 hidden_state
 * 
 * Shared data:
 * - KV cache: Per-layer managed by LLMKVCacheManager
 * - ROPE (cos/sin): Shared from Shard 0 output across all shards
 * - Hidden state: Chained between shards (Shard N output → Shard N+1 input)
 * - Attention mask: Shared buffer broadcasted to all shards
 */

#include "llm_decode_runner.h"
#include "llm_input_preparer.h"
#include "qnn_tensor_util.h"
#include "binary_provider.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

namespace llama_qnn {

bool LLMDecodeRunner::load_multi_context_graphs() { // [spagetti] blob 고려 / et에서 cache가 뭔지 왜 필요한지 확인해봐야함
  // Auto-detect or use specified num_shards
  std::vector<std::string> context_files;
  std::vector<std::string> json_files;
  
  if (config_.num_shards == 0) {
    // Auto-detect: scan for forward_0.bin, forward_1.bin, ...
    std::cout << "[Multi-Context] Auto-detecting shards in: " << config_.ctx_dir << "\n";
    
    for (int i = 0; i < 100; ++i) {  // Max 100 shards
      std::string ctx_file = config_.ctx_dir + "/forward_" + std::to_string(i) + ".bin";
      std::string json_file = config_.ctx_dir + "/forward_" + std::to_string(i) + "_json.json";
      
      std::ifstream ctx_test(ctx_file);
      std::ifstream json_test(json_file);
      
      bool ctx_exists = ctx_test.good();
      bool json_exists = json_test.good();
      
      std::cout << "[Multi-Context] Shard " << i << ": "
                << "bin=" << (ctx_exists ? "✓" : "✗") << " "
                << "json=" << (json_exists ? "✓" : "✗");
      
      if (!ctx_exists || !json_exists) {
        std::cout << " → stopping\n";
        if (i == 0) {
          std::cout << "[Multi-Context] ⚠ No shards found! Tried:\n"
                    << "  " << ctx_file << "\n"
                    << "  " << json_file << "\n";
        }
        break;  // No more shards found
      }
      
      std::cout << " → found\n";
      context_files.push_back(ctx_file);
      json_files.push_back(json_file);
    }
    
    config_.num_shards = context_files.size();
    
    if (config_.num_shards == 0) {
      error_msg_ = "No context files found in " + config_.ctx_dir;
      return false;
    }
    
    std::cout << "[Multi-Context] Total shards detected: " << config_.num_shards << "\n";
  } else {
    // Use specified num_shards
    for (int i = 0; i < config_.num_shards; ++i) {
      std::string ctx_file = config_.ctx_dir + "/forward_" + std::to_string(i) + ".bin";
      std::string json_file = config_.ctx_dir + "/forward_" + std::to_string(i) + "_json.json";
      
      std::ifstream ctx_test(ctx_file);
      std::ifstream json_test(json_file);
      
      if (!ctx_test || !json_test) {
        error_msg_ = "Missing context or JSON file for shard " + std::to_string(i);
        return false;
      }
      
      context_files.push_back(ctx_file);
      json_files.push_back(json_file);
    }
  }
  
  if (config_.log_level >= 1) {
    std::cout << "[Multi-Context] Found " << config_.num_shards << " shards\n";
  }
  
  // Load each shard ONE BY ONE (메모리 절약: 읽기→생성→해제 반복)
  for (int i = 0; i < config_.num_shards; ++i) {
    if (config_.log_level >= 1) {
      std::cout << "[Graphs] Shard " << i << ": loading...\n";
    }
    
    // Read binary file (한 번에 하나만)
    std::ifstream ifs(context_files[i], std::ios::binary | std::ios::ate);
    if (!ifs) {
      error_msg_ = "Failed to open context binary: " + context_files[i];
      return false;
    }
    
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    
    if (!ifs.read(buffer.data(), size)) {
      error_msg_ = "Failed to read context binary: " + context_files[i];
      return false;
    }
    ifs.close();
    
    // Create context from binary (하나만 생성)
    if (!loader_->create_context_from_binary(buffer.data(), size)) {
      error_msg_ = "Failed to create context from binary: " + context_files[i];
      return false;
    }
    
    if (config_.log_level >= 1) {
      std::cout << "[Graphs] Shard " << i << ": context created\n";
    }
    
    // buffer는 스코프를 벗어나면서 자동으로 메모리 해제됨
  }
  
  if (config_.log_level >= 1) {
    std::cout << "[Multi-Context] All " << loader_->num_contexts() << " contexts created\n";
  }
  
  // Parse JSON and retrieve graphs for each shard
  shards_.resize(config_.num_shards);
  
  for (int i = 0; i < config_.num_shards; ++i) {
    if (!parse_qnn_json(json_files[i], shards_[i].graphs)) {
      error_msg_ = "Failed to parse JSON: " + json_files[i];
      return false;
    }
    
    if (shards_[i].graphs.find("prefill_forward") == shards_[i].graphs.end() ||
        shards_[i].graphs.find("kv_forward") == shards_[i].graphs.end()) {
      error_msg_ = "Required graphs not found in shard " + std::to_string(i);
      return false;
    }
    
    shards_[i].prefill_graph = &shards_[i].graphs["prefill_forward"];
    shards_[i].kv_graph = &shards_[i].graphs["kv_forward"];
    
    // Retrieve graphs
    if (!loader_->retrieve_graph(i, "prefill_forward") ||
        !loader_->retrieve_graph(i, "kv_forward")) {
      error_msg_ = "Failed to retrieve graphs for shard " + std::to_string(i);
      return false;
    }
    
    if (config_.log_level >= 2) {
      std::cout << "[Shard " << i << "] prefill_forward: "
                << shards_[i].prefill_graph->inputs.size() << " inputs, "
                << shards_[i].prefill_graph->outputs.size() << " outputs\n";
      std::cout << "[Shard " << i << "] kv_forward: "
                << shards_[i].kv_graph->inputs.size() << " inputs, "
                << shards_[i].kv_graph->outputs.size() << " outputs\n";
    }
  }
  
  return true;
}

bool LLMDecodeRunner::extract_multi_context_metadata() {
  // 1. Extract dimensions from shard 0 graph (always needed from attention mask)
  auto* shard0_prefill = shards_[0].prefill_graph;
  context_len_ = 0;
  prefill_ar_len_ = 0;
  kv_ar_len_ = 0;
  
  // Find attention mask to get context_len and ar_len
  for (const auto& t : shard0_prefill->inputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    
    if (name_lower.find("atten_mask") != std::string::npos && t.dims.size() >= 2) {
      prefill_ar_len_ = t.dims[t.dims.size() - 2];
      context_len_ = t.dims[t.dims.size() - 1];
      break;
    }
  }
  
  // Extract kv_ar_len from kv_forward
  for (const auto& t : shards_[0].kv_graph->inputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    
    if (name_lower.find("atten_mask") != std::string::npos && t.dims.size() >= 2) {
      kv_ar_len_ = t.dims[t.dims.size() - 2];
      break;
    }
  }
  
  // 2. Use params.json if available, otherwise infer from graph
  if (model_params_.is_valid()) {
    // Use values from params.json
    num_layers_ = model_params_.n_layers;
    num_heads_ = model_params_.n_kv_heads;
    head_dim_ = model_params_.head_dim;
    layers_per_shard_ = (config_.num_shards > 0) ? 
                        (num_layers_ / config_.num_shards) : 0;
  } else {
    // Fallback: extract from shard 0 graph tensors
    head_dim_ = 0;
    
    // Extract head_dim from KV cache tensor
    for (const auto& t : shard0_prefill->inputs) {
      if (t.name.find("_args_") != std::string::npos && t.dims.size() == 3) {
        head_dim_ = t.dims[2];
        if (head_dim_ > 0) break;
      }
    }
    
    // Count KV cache tensors to infer layers_per_shard and num_heads
    int kv_count = 0;
    for (const auto& t : shard0_prefill->inputs) {
      if (t.name.find("_args_") != std::string::npos && t.dims.size() == 3) {
        kv_count++;
      }
    }
    
    // Infer: kv_count = layers_per_shard * 2 (K+V) * num_heads
    num_layers_ = 16;  // Default: 16 layers
    layers_per_shard_ = (config_.num_shards > 0) ? 
                        (num_layers_ / config_.num_shards) : 2;
    num_heads_ = kv_count / (layers_per_shard_ * 2);
  }
  
  prefill_cache_len_ = context_len_ - prefill_ar_len_;
  kv_cache_len_ = context_len_ - kv_ar_len_;
  
  if (config_.log_level >= 1) {
    std::cout << "[Multi-Context Metadata]\n";
    std::cout << "  context_len=" << context_len_ << ", prefill_ar=" << prefill_ar_len_
              << ", kv_ar=" << kv_ar_len_ << "\n";
    std::cout << "  num_layers=" << num_layers_ << ", num_heads=" << num_heads_
              << ", head_dim=" << head_dim_ << "\n";
    std::cout << "  num_shards=" << config_.num_shards << ", layers_per_shard=" << layers_per_shard_ << "\n";
    if (model_params_.is_valid()) {
      std::cout << "  Source: params.json ✓\n";
    } else {
      std::cout << "  Source: inferred from graph tensors\n";
    }
  }
  
  if (context_len_ == 0 || num_heads_ == 0 || head_dim_ == 0) {
    error_msg_ = "Failed to extract valid metadata";
    return false;
  }
  
  return true;
}

bool LLMDecodeRunner::setup_multi_context_kv_cache() {
  // Setup KV cache (same as single-context)
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
              << " MiB (layer-wise for all shards)\n";
  }
  
  return true;
}

bool LLMDecodeRunner::setup_multi_context_io_allocators() { // [spagetti] 이거 문제가 많다 전체 재설계 해야할 수도 있음
  for (int i = 0; i < config_.num_shards; ++i) {
    auto& shard = shards_[i];
    
    // Prefill allocator
    shard.prefill_alloc.reset(new QNNIOAllocator());
    shard.prefill_alloc->build_from_qnnjson(*shard.prefill_graph);
    auto prefill_bytes = shard.prefill_alloc->allocate(0); // [spagetti] 이거 지금 말안됨 근거 없는 코드임 내가 돌아가게만 만든 코드
    
    // // KV allocator
    shard.kv_alloc.reset(new QNNIOAllocator());
    shard.kv_alloc->build_from_qnnjson(*shard.kv_graph);
    auto kv_bytes = shard.kv_alloc->allocate(64); // [spagetti] 이거 왜 두번 할당함? 그럼 실제로는 뭘씀? 

    // Create tensor holders for prefill graph
    shard.prefill_input_holders.clear();
    for (const auto& t : shard.prefill_graph->inputs) { // [spagetti] 모든 인풋에 대해서 이렇게 만드는게 맞나? 확인
      auto h = std::make_unique<QnnTensorHolder>();
      h->init_from_json(t, nullptr, t.nbytes, true);
      shard.prefill_input_holders.push_back(std::move(h));
    }
    
    shard.prefill_output_holders.clear();
    for (const auto& t : shard.prefill_graph->outputs) {
      auto h = std::make_unique<QnnTensorHolder>();
      h->init_from_json(t, nullptr, t.nbytes, false);
      shard.prefill_output_holders.push_back(std::move(h));
    }
    
    // Create tensor holders for kv graph
    shard.kv_input_holders.clear();
    for (const auto& t : shard.kv_graph->inputs) {
      auto h = std::make_unique<QnnTensorHolder>();
      h->init_from_json(t, nullptr, t.nbytes, true);
      shard.kv_input_holders.push_back(std::move(h));
    }
    
    shard.kv_output_holders.clear();
    for (const auto& t : shard.kv_graph->outputs) {
      auto h = std::make_unique<QnnTensorHolder>();
      h->init_from_json(t, nullptr, t.nbytes, false);
      shard.kv_output_holders.push_back(std::move(h));
    }
    
    if (config_.log_level >= 2) {
      std::cout << "[Shard " << i << " I/O] Prefill: " << (prefill_bytes / 1024.0)
                << " KiB, Decode: " << (kv_bytes / 1024.0) << " KiB\n";
      std::cout << "[Shard " << i << " Holders] Prefill I/O: "
                << shard.prefill_input_holders.size() << "/"
                << shard.prefill_output_holders.size() << ", KV I/O: "
                << shard.kv_input_holders.size() << "/"
                << shard.kv_output_holders.size() << "\n";
    }
  }
  
  return true;
}

bool LLMDecodeRunner::allocate_shared_buffers() {
  // Allocate shared buffers for data that needs to be passed between shards
  
  // 1. Hidden state: [1, ar_len, dim] for prefill, [1, 1, dim] for decode
  // Use prefill_ar_len for max size (e.g., [1, 32, 2048])
  int hidden_dim = model_params_.is_valid() ? model_params_.dim : 2048;
  size_t hidden_state_size = prefill_ar_len_ * hidden_dim * sizeof(uint16_t);  // UFIXED_16
  void* hidden_state_buf = aligned_alloc(64, hidden_state_size);
  if (!hidden_state_buf) {
    error_msg_ = "Failed to allocate hidden state buffer";
    return false;
  }
  std::memset(hidden_state_buf, 0, hidden_state_size);
  shared_buffer_views_["hidden_state"] = hidden_state_buf;
  
  // 2. ROPE cos/sin: [max_seq_len, head_dim/2] - typically from shard 0 output
  // Size depends on context_len, allocate generously
  size_t rope_size = context_len_ * head_dim_ * sizeof(uint16_t);
  void* rope_cos_buf = aligned_alloc(64, rope_size);
  void* rope_sin_buf = aligned_alloc(64, rope_size);
  if (!rope_cos_buf || !rope_sin_buf) {
    error_msg_ = "Failed to allocate ROPE buffers";
    return false;
  }
  std::memset(rope_cos_buf, 0, rope_size);
  std::memset(rope_sin_buf, 0, rope_size);
  shared_buffer_views_["rope_cos"] = rope_cos_buf;
  shared_buffer_views_["rope_sin"] = rope_sin_buf;
  
  // 3. Attention mask: [ar_len, context_len]
  size_t attn_mask_size = prefill_ar_len_ * context_len_ * sizeof(uint16_t);
  void* attn_mask_buf = aligned_alloc(64, attn_mask_size);
  if (!attn_mask_buf) {
    error_msg_ = "Failed to allocate attention mask buffer";
    return false;
  }
  std::memset(attn_mask_buf, 0, attn_mask_size);
  shared_buffer_views_["attention_mask"] = attn_mask_buf;
  
  if (config_.log_level >= 1) {
    std::cout << "[Shared Buffers] Allocated:\n";
    std::cout << "  hidden_state: " << (hidden_state_size / 1024.0) << " KiB\n";
    std::cout << "  rope_cos/sin: " << (2 * rope_size / 1024.0) << " KiB\n";
    std::cout << "  attention_mask: " << (attn_mask_size / 1024.0) << " KiB\n";
  }
  
  return true;
}

bool LLMDecodeRunner::run_multi_context_prefill(llama_context * ctx, llama_batch batch) {
  // Extract tokens from batch
  std::vector<int32_t> tokens(batch.n_tokens);
  for (int i = 0; i < batch.n_tokens; ++i) {
    tokens[i] = batch.token[i];
  }
  
  if (config_.log_level >= 1) {
    std::cout << "[Multi-Context Prefill] Starting with " << tokens.size() << " tokens\n";
  }
  
  int32_t n_past = 0;
  int32_t num_tokens = tokens.size();
  uint16_t* attn_mask = reinterpret_cast<uint16_t*>(shared_buffer_views_["attention_mask"]);


  if (config_.log_level >= 1) {
    std::cout << "[Multi-Context Prefill] Input tokens:\n";
    for (size_t i = 0; i < tokens.size(); ++i) {
      std::cout << "  token[" << i << "] = " << tokens[i] << "\n";
    }
  }
  
  // Multiple iteration prefill
  while (n_past < num_tokens) {
    int32_t chunk_size = std::min(prefill_ar_len_, num_tokens - n_past);
    
    if (config_.log_level >= 2) {
      std::cout << "[Multi-Context Prefill] Iteration: n_past=" << n_past 
                << ", chunk_size=" << chunk_size << "\n";
    }
    
    // Extract current chunk of tokens
    std::vector<int32_t> chunk_tokens(
      tokens.begin() + n_past,
      tokens.begin() + n_past + chunk_size
    );

    // Pad chunk to prefill_ar_len if needed
    if (chunk_size < prefill_ar_len_) {
      chunk_tokens.resize(prefill_ar_len_, 0);  // Pad with 0
    }
    
    // Update attention mask for this iteration
    // SMART_MASK: causal pattern within the current chunk, plus attending to all past tokens
    std::memset(attn_mask, 0, prefill_ar_len_ * context_len_ * sizeof(uint16_t));
    
    for (int i = 0; i < chunk_size; ++i) {
      int row = i;
      // Attend to all past tokens [0..n_past-1]
      for (int j = 0; j < n_past; ++j) {
        attn_mask[row * context_len_ + j] = 65535;
      }
      // Attend to current and previous tokens in this chunk (causal)
      int chunk_start = context_len_ - prefill_ar_len_;
      for (int j = 0; j <= i; ++j) {
        attn_mask[row * context_len_ + chunk_start + j] = 65535;
      }
    }
    
    // Run prefill through all shards sequentially
    for (int shard_idx = 0; shard_idx < config_.num_shards; ++shard_idx) {
      if (!run_shard_prefill(shard_idx, chunk_tokens, n_past, chunk_size)) {
        return false;
      }
    }
    
    // Update KV cache: copy prefill outputs to KV inputs for this iteration
    if (config_.log_level >= 2) {
      std::cout << "[Multi-Context Prefill] Updating KV cache for iteration at n_past=" 
                << n_past << ", chunk_size=" << chunk_size << "\n";
    }
    
    int total_v_updated = 0, total_k_updated = 0;
    
    for (int shard_idx = 0; shard_idx < config_.num_shards; ++shard_idx) {
      auto& shard = shards_[shard_idx];
      auto& bindings = shard.prefill_alloc->bindings();
      int shard_layer_base = shard_idx * layers_per_shard_;
      
      // Collect V and K cache outputs (in original order, no sorting!)
      std::vector<void*> v_outputs, k_outputs;
      for (const auto& t : shard.prefill_graph->outputs) {
        auto bit = bindings.find(t.name);
        if (bit == bindings.end()) continue;
        
        if (t.name.find("output_aten_view_copy_default_") != std::string::npos) {
          v_outputs.push_back(bit->second);
        } else if (t.name.find("output_aten_permute_copy_default_") != std::string::npos) {
          k_outputs.push_back(bit->second);
        }
      }
      
      // Process V caches
      for (size_t i = 0; i < v_outputs.size(); ++i) {
        int local_layer = i / num_heads_;
        int head = i % num_heads_;
        int global_layer = shard_layer_base + local_layer;
        
        if (global_layer >= num_layers_) continue;
        
        const auto& v_buf = kv_manager_->get_v_cache(global_layer, head);
        uint8_t* src = reinterpret_cast<uint8_t*>(v_outputs[i]);
        uint8_t* dst = reinterpret_cast<uint8_t*>(v_buf.input_buffer) + n_past * head_dim_;
        std::memcpy(dst, src, chunk_size * head_dim_);
        
        if (config_.log_level >= 2 && shard_idx == 0 && i < 2) {
          std::cout << "[Prefill KV] Iter: n_past=" << n_past << " Shard " << shard_idx 
                    << " V-cache " << i << " → Layer " << global_layer << " Head " << head << "\n";
        }
        total_v_updated++;
      }
      
      // Process K caches
      for (size_t i = 0; i < k_outputs.size(); ++i) {
        int local_layer = i / num_heads_;
        int head = i % num_heads_;
        int global_layer = shard_layer_base + local_layer;
        
        if (global_layer >= num_layers_) continue;
        
        const auto& k_buf = kv_manager_->get_k_cache(global_layer, head);
        uint8_t* src = reinterpret_cast<uint8_t*>(k_outputs[i]);
        uint8_t* dst = reinterpret_cast<uint8_t*>(k_buf.input_buffer) + n_past;
        
        // K cache: copy with stride (transposed layout)
        for (int32_t dim = 0; dim < head_dim_; ++dim) {
          std::memcpy(dst, src, chunk_size);
          src += prefill_ar_len_;
          dst += prefill_cache_len_;
        }
        total_k_updated++;
      }
    }
    
    if (config_.log_level >= 3) {
      std::cout << "[Multi-Context Prefill] KV cache updated: "
                << total_v_updated << " V-caches, "
                << total_k_updated << " K-caches\n";
    }
    
    // Advance n_past for next iteration
    n_past += chunk_size;
  }  // End of while loop
  
  if (config_.log_level >= 2) {
    std::cout << "[Multi-Context Prefill] All iterations completed. Total tokens processed: " 
              << n_past << "\n";
  }
  
  // Extract logits from final shard
  int final_shard = config_.num_shards - 1;
  const QnnJsonTensorDesc* logits_desc = nullptr;
  
  // Find logits output
  for (const auto& t : shards_[final_shard].prefill_graph->outputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    
    if (name_lower.find("squeeze") != std::string::npos) {
      logits_desc = &t;
      break;
    }
  }
  
  // Fallback: use the largest output tensor
  if (!logits_desc) { // [spagetti] 이 부분 뭐하는거임? 필요한거임?
    size_t max_size = 0;
    for (const auto& t : shards_[final_shard].prefill_graph->outputs) {
      if (t.dims.size() == 3 && t.name.find("_args_") != std::string::npos) {
        continue;
      }
      if (t.nbytes > max_size) {
        max_size = t.nbytes;
        logits_desc = &t;
      }
    }
  }
  
  if (!logits_desc) {
    error_msg_ = "Logits output not found in final shard";
    return false;
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Multi-Context Prefill] Logits tensor: " << logits_desc->name 
              << " (" << logits_desc->nbytes << " bytes)\n";
  }
  
  // Get logits buffer
  auto& bindings = shards_[final_shard].prefill_alloc->bindings();
  auto it = bindings.find(logits_desc->name);
  if (it == bindings.end()) {
    error_msg_ = "Logits buffer not found in bindings";
    return false;
  }
  
  if (!it->second) {
    error_msg_ = "Logits buffer is null";
    return false;
  }

  const uint16_t * logits_u16 = reinterpret_cast<const uint16_t *>(it->second);

  // Determine vocab size: prefer model_params_ if available, otherwise tensor dims
  int32_t vocab_size = model_params_.is_valid() ? model_params_.vocab_size : 0;
  if (vocab_size <= 0 && !logits_desc->dims.empty()) {
    vocab_size = (int32_t) logits_desc->dims.back();
  }
  if (vocab_size <= 0) {
    error_msg_ = "Invalid vocab size for multi-context prefill logits";
    return false;
  }

  // For prefill, logits are [batch=1, prefill_ar_len, vocab_size]
  // We want the last token's logits from the last iteration
  // If we processed 50 tokens with prefill_ar_len=32:
  //   - Iteration 1: tokens[0:32] (32 tokens)
  //   - Iteration 2: tokens[32:50] (18 tokens)
  // The last iteration's output contains 18 tokens worth of logits,
  // and we want the last one (index 17 in that chunk)
  int32_t last_chunk_size = ((num_tokens - 1) % prefill_ar_len_) + 1;
  int32_t last_token_index = last_chunk_size - 1;

  if (last_token_index < 0) {
    error_msg_ = "Invalid last token index in multi-context prefill";
    return false;
  }

  size_t last_token_offset = (size_t) last_token_index * (size_t) vocab_size;

  // Update n_past_: total tokens processed
  n_past_ = num_tokens;
  
  // // Original greedy argmax on uint16_t logits
  // uint16_t max_val = logits_u16[last_token_offset];
  // next_token = 0;
  // for (int32_t i = 1; i < vocab_size; ++i) {
  //   if (logits_u16[last_token_offset + i] > max_val) {
  //     max_val = logits_u16[last_token_offset + i];
  //     next_token = i;
  //   }
  // }

  // Inject logits into llama_context for external sampling if provided
  if (ctx != nullptr) {
    std::vector<float> last_row_f32(vocab_size);

    // Use quantization parameters if they are sane; otherwise fall back to simple cast
    float scale = logits_desc->quant_scale;
    int32_t offset = logits_desc->quant_offset;
    GGML_ASSERT(scale != 0.0f && "Quantization scale should not be zero");

    for (int32_t i = 0; i < vocab_size; ++i) {
      float q = (float) logits_u16[last_token_offset + i];
      last_row_f32[i] = (q + offset) * scale;
    }

    llama_set_logits_external(ctx, last_row_f32.data(), 1);
  }
  
  return true;
}

bool LLMDecodeRunner::run_multi_context_decode_step(llama_context * ctx, llama_batch batch) {
  // Extract token from batch
  if (batch.n_tokens != 1) {
    error_msg_ = "Decode step expects exactly 1 token in batch";
    return false;
  }
  int32_t token_in = batch.token[0];
  
  // Rearrange cache on first decode (prefill→decode transition)
  if (!prefill_done_) {
    if (config_.log_level >= 1) {
      std::cout << "[Multi-Context Decode] First decode - rearranging cache\n";
    }
    kv_manager_->rearrange_cache(prefill_ar_len_, kv_ar_len_);
    prefill_done_ = true;
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Multi-Context Decode] Step: token=" << token_in 
              << ", n_past=" << n_past_ << "\n";
  }
  
  // Prepare shard 0 inputs: token, position, attention_mask
  auto& shard0 = shards_[0];
  auto& bindings0 = shard0.kv_alloc->bindings();
  
  for (const auto& t : shard0.kv_graph->inputs) {
    auto it = bindings0.find(t.name);
    if (it == bindings0.end()) continue;
    
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    
    // Token
    if (name_lower.find("token") != std::string::npos && name_lower.find("input") != std::string::npos) {
      std::memcpy(it->second, &token_in, sizeof(int32_t));
      if (config_.log_level >= 2) {
        std::cout << "[Decode Shard 0] Token filled: " << token_in << "\n";
      }
    }
    // Position
    else if (name_lower.find("pos") != std::string::npos && t.data_type.find("INT_32") != std::string::npos) {
      std::memcpy(it->second, &n_past_, sizeof(int32_t));
      if (config_.log_level >= 2) {
        std::cout << "[Decode Shard 0] Position filled: " << n_past_ << "\n";
      }
    }
    // Attention mask
    else if (name_lower.find("atten_mask") != std::string::npos) {
      uint16_t* attn_mask = reinterpret_cast<uint16_t*>(it->second);
      std::memset(attn_mask, 0, context_len_ * sizeof(uint16_t));
      
      // Attend to past tokens [0..n_past_-1]
      for (int32_t i = 0; i < n_past_; ++i) {
        attn_mask[i] = 65535;
      }
      // Attend to current token (last position in rearranged cache)
      attn_mask[context_len_ - 1] = 65535;
      
      if (config_.log_level >= 2) {
        std::cout << "[Decode Shard 0] Attention mask: attend to [0, " << (n_past_ - 1) << "] and [" << (context_len_ - 1) << "] (" << (n_past_ + 1) << " tokens)\n";
      }
      
      // Also copy to shared buffer for other shards
      std::memcpy(shared_buffer_views_["attention_mask"], attn_mask, context_len_ * sizeof(uint16_t));
    }
  }
  
  // Run decode through all shards sequentially
  for (int shard_idx = 0; shard_idx < config_.num_shards; ++shard_idx) {
    if (config_.log_level >= 2) {
      std::cout << "[Multi-Context Decode] Running shard " << shard_idx << "...\n";
    }
    if (!run_shard_decode(shard_idx, n_past_)) {
      return false;
    }
    if (config_.log_level >= 2) {
      std::cout << "[Multi-Context Decode] Shard " << shard_idx << " completed\n";
    }
  }
  
  // Update KV cache: copy decode outputs to inputs for next step
  // Manual memcpy (exactly like single-context)
  if (config_.log_level >= 2) {
    std::cout << "[Multi-Context Decode] Updating KV cache at position " << n_past_ << "...\n";
  }
  
  int total_v_updated = 0, total_k_updated = 0;
  
  // Copy KV cache outputs to inputs for all shards
  for (int shard_idx = 0; shard_idx < config_.num_shards; ++shard_idx) {
    auto& shard = shards_[shard_idx];
    auto& bindings = shard.kv_alloc->bindings();
    int shard_layer_base = shard_idx * layers_per_shard_;
    
    // Collect V and K cache outputs (in original order, no sorting!)
    std::vector<void*> v_outputs, k_outputs;
    for (const auto& t : shard.kv_graph->outputs) {
      auto bit = bindings.find(t.name);
      if (bit == bindings.end()) continue;
      
      if (t.name.find("output_aten_view_copy_default_") != std::string::npos) {
        v_outputs.push_back(bit->second);
      } else if (t.name.find("output_aten_permute_copy_default_") != std::string::npos) {
        k_outputs.push_back(bit->second);
      }
    }
    
    // Process V caches (순서대로 layer/head 할당)
    for (size_t i = 0; i < v_outputs.size(); ++i) {
      int local_layer = i / num_heads_;
      int head = i % num_heads_;
      int global_layer = shard_layer_base + local_layer;
      
      if (global_layer >= num_layers_) continue;
      
      const auto& v_buf = kv_manager_->get_v_cache(global_layer, head);
      uint8_t* src = reinterpret_cast<uint8_t*>(v_outputs[i]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(v_buf.input_buffer) + n_past_ * head_dim_;
      std::memcpy(dst, src, 1 * head_dim_);
      total_v_updated++;
    }
    
    // Process K caches (순서대로 layer/head 할당)
    for (size_t i = 0; i < k_outputs.size(); ++i) {
      int local_layer = i / num_heads_;
      int head = i % num_heads_;
      int global_layer = shard_layer_base + local_layer;
      
      if (global_layer >= num_layers_) continue;
      
      const auto& k_buf = kv_manager_->get_k_cache(global_layer, head);
      uint8_t* src = reinterpret_cast<uint8_t*>(k_outputs[i]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(k_buf.input_buffer) + n_past_;
      
      // K cache: copy with stride (transposed layout)
      for (int32_t dim = 0; dim < head_dim_; ++dim) {
        dst[dim * kv_cache_len_] = src[dim];
      }
      total_k_updated++;
    }
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Multi-Context Decode] KV cache updated: " 
              << total_v_updated << " V-caches, " 
              << total_k_updated << " K-caches\n";
  }
  
  // Extract logits from final shard (kv_forward)
  int final_shard = config_.num_shards - 1;
  const QnnJsonTensorDesc* logits_desc = nullptr;
  
  for (const auto& t : shards_[final_shard].kv_graph->outputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    
    if (name_lower.find("squeeze") != std::string::npos ||
        name_lower.find("logit") != std::string::npos ||
        name_lower.find("lm_head") != std::string::npos) {
      logits_desc = &t;
      break;
    }
  }
  
  // Fallback: use the largest output tensor
  if (!logits_desc) {
    size_t max_size = 0;
    for (const auto& t : shards_[final_shard].kv_graph->outputs) {
      if (t.dims.size() == 3 && t.name.find("_args_") != std::string::npos) {
        continue;
      }
      if (t.nbytes > max_size) {
        max_size = t.nbytes;
        logits_desc = &t;
      }
    }
  }
  
  if (!logits_desc) {
    error_msg_ = "Logits output not found in final shard";
    return false;
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Multi-Context Decode] Using logits from: " << logits_desc->name << "\n";
  }
  
  auto& bindings = shards_[final_shard].kv_alloc->bindings();
  
  // Debug: Print first few logits values
  if (config_.log_level >= 2) {
    auto it_debug = bindings.find(logits_desc->name);
    if (it_debug != bindings.end()) {
      const uint16_t* logits_debug = reinterpret_cast<const uint16_t*>(it_debug->second);
      std::cout << "[Decode Logits] First 8 values: ";
      for (int i = 0; i < 8; ++i) {
        std::cout << logits_debug[i] << " ";
      }
      std::cout << "\n";
    }
  }
  auto it = bindings.find(logits_desc->name);
  if (it == bindings.end()) {
    error_msg_ = "Logits buffer not found";
    return false;
  }

  if (!it->second) {
    error_msg_ = "Logits buffer is null";
    return false;
  }

  const uint16_t * logits_u16 = reinterpret_cast<const uint16_t *>(it->second);

  // Determine vocab size
  int32_t vocab_size = model_params_.is_valid() ? model_params_.vocab_size : 0;
  if (vocab_size <= 0 && !logits_desc->dims.empty()) {
    vocab_size = (int32_t) logits_desc->dims.back();
  }
  if (vocab_size <= 0) {
    error_msg_ = "Invalid vocab size for multi-context decode logits";
    return false;
  }

  // // Original greedy argmax on uint16_t logits
  // uint16_t max_val = logits_u16[0];
  // token_out = 0;
  // for (int32_t i = 1; i < vocab_size; ++i) {
  //   if (logits_u16[i] > max_val) {
  //     max_val = logits_u16[i];
  //     token_out = i;
  //   }
  // }

  // Inject logits into llama_context for external sampling if provided
  if (ctx != nullptr) {
    std::vector<float> row_f32(vocab_size);

    float scale = logits_desc->quant_scale;
    int32_t offset = logits_desc->quant_offset;
    GGML_ASSERT(scale != 0.0f && "Quantization scale should not be zero");

    for (int32_t i = 0; i < vocab_size; ++i) {
      float q = (float) logits_u16[i];
      row_f32[i] = (q + offset) * scale;
    }

    llama_set_logits_external(ctx, row_f32.data(), 1);
  }
  
  // Update n_past_ for next decode step
  n_past_ += 1;
  
  return true;
}

bool LLMDecodeRunner::run_shard_prefill(int shard_idx,
                                         const std::vector<int32_t>& tokens,
                                         int32_t n_past,
                                         int32_t n_update) {
  if (config_.log_level >= 1) {
    std::cout << "[Shard " << shard_idx << " Prefill] Running...\n";
  }
  
  auto& shard = shards_[shard_idx];
  auto& bindings = shard.prefill_alloc->bindings();
  
  // Setup KV cache override for this shard's layers
  std::map<std::string, void*> kv_override;
  int shard_layer_start = shard_idx * layers_per_shard_;
  int shard_layer_end = shard_layer_start + layers_per_shard_;
  
  if (config_.log_level >= 2) {
    std::cout << "[Shard " << shard_idx << "] Layer range: " << shard_layer_start 
              << "-" << (shard_layer_end - 1) << "\n";
  }
  
  // Count KV cache tensors for this shard
  // Input KV cache: dims[1] tells us if it's V cache (small value like 32, 992, 1023) 
  // or K cache (larger value like 64)
  int v_count = 0, k_count = 0;
  for (const auto& t : shard.prefill_graph->inputs) {
    if (t.name.find("_args_") != std::string::npos && t.dims.size() == 3) {
      // V cache: [1, cache_len, head_dim=64]
      // K cache: [1, head_dim=64, cache_len]
      bool is_v = (t.dims[2] == head_dim_);  // V: last dim is head_dim
      bool is_k = (t.dims[1] == head_dim_);  // K: middle dim is head_dim
      
      if (is_v || is_k) {
        // This shard's local index
        int local_idx = is_v ? v_count++ : k_count++;
        int local_layer = local_idx / num_heads_;
        int head = local_idx % num_heads_;
        
        // Convert to global layer index
        int global_layer = shard_layer_start + local_layer;
        
        if (global_layer < num_layers_) {
          const auto& cache_buf = is_v ? 
            kv_manager_->get_v_cache(global_layer, head) :
            kv_manager_->get_k_cache(global_layer, head);
          kv_override[t.name] = cache_buf.input_buffer;
        } else if (config_.log_level >= 2) {
          std::cout << "[Shard " << shard_idx << "] WARNING: global_layer " << global_layer 
                    << " >= num_layers " << num_layers_ << "\n";
        }
      }
    }
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Shard " << shard_idx << "] KV cache bound: " << kv_override.size() << " tensors\n";
  }
  
  // 1. Fill input buffers
  if (shard_idx == 0) {
    if (config_.log_level >= 2) {
      std::cout << "[Shard 0] Filling inputs: " << tokens.size() << " tokens\n";
      std::cout << "[Shard 0] First token: " << tokens[0] << "\n";
    }
    
    // Fill tokens and positions (skip attention mask - we manage it manually)
    InputPreparer::auto_fill_inputs(*shard.prefill_graph,
      [&](const std::string& name) -> void* {
        // Check KV override first
        auto ko = kv_override.find(name);
        if (ko != kv_override.end()) return ko->second;
        
        // Otherwise use normal binding
        auto it = bindings.find(name);
        return (it != bindings.end()) ? it->second : nullptr;
      },
      tokens,
      n_past,  // start_pos: use current n_past for position tensor
      true,    // skip_attention_mask: we manually set it before this call
      config_.log_level >= 2);
    
    // Copy manually-set attention mask to shard 0's input buffer
    for (const auto& t : shard.prefill_graph->inputs) {
      std::string name_lower = t.name;
      for (auto& c : name_lower) c = (char)tolower(c);
      if (name_lower.find("atten_mask") != std::string::npos) {
        auto it = bindings.find(t.name);
        if (it != bindings.end()) {
          std::memcpy(it->second, shared_buffer_views_["attention_mask"], t.nbytes);
          if (config_.log_level >= 2) {
            std::cout << "[Shard 0] Attention mask copied from shared buffer\n";
          }
        }
        break;
      }
    }
  } else {
    // Shard 1-7: hidden_state, ROPE, attention_mask
    if (config_.log_level >= 2) {
      std::cout << "[Shard " << shard_idx << "] Copying from shared buffers...\n";
      if (shard_idx == 1) {
        std::cout << "[Shard 1] All non-KV input names:\n";
        for (const auto& t : shard.prefill_graph->inputs) {
          auto ko = kv_override.find(t.name);
          if (ko == kv_override.end()) {
            std::cout << "  - " << t.name << "\n";
          }
        }
      }
    }
    
    // Copy inputs from shared buffers
    for (const auto& t : shard.prefill_graph->inputs) { // [spagetti] 정확하게 어떻게 동작하는건지 확인할 필요 있음
      auto ko = kv_override.find(t.name);
      if (ko != kv_override.end()) continue;  // Skip KV cache (already bound)
      
      auto it = bindings.find(t.name);
      if (it == bindings.end()) continue;
      
      std::string name_lower = t.name;
      for (auto& c : name_lower) c = (char)tolower(c);
      
      // Hidden state
      if (name_lower.find("fallback") != std::string::npos) {
        std::memcpy(it->second, shared_buffer_views_["hidden_state"], t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Shard " << shard_idx << "] Hidden state copied: " << t.nbytes << " bytes\n";
        }
      }
      // ROPE cos
      else if (t.name.find("input_9_aten_view_copy_default_0") != std::string::npos || t.name.find("input_9_aten_select_copy_int_0") != std::string::npos) {
        std::memcpy(it->second, shared_buffer_views_["rope_cos"], t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Shard " << shard_idx << "] ROPE cos copied: " << t.nbytes << " bytes\n";
        }
      }
      // ROPE sin
      else if (t.name.find("input_10_aten_view_copy_default_1_0") != std::string::npos || t.name.find("input_10_aten_select_copy_int_1_0") != std::string::npos) {
        std::memcpy(it->second, shared_buffer_views_["rope_sin"], t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Shard " << shard_idx << "] ROPE sin copied: " << t.nbytes << " bytes\n";
        }
      }
      // Attention mask
      else if (name_lower.find("atten_mask") != std::string::npos) {
        std::memcpy(it->second, shared_buffer_views_["attention_mask"], t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Shard " << shard_idx << "] Attention mask copied: " << t.nbytes << " bytes\n";
        }
      }
    }
  }
  
  // 2. Build tensor lists and execute
  std::vector<Qnn_Tensor_t> inputs, outputs;
  
  for (size_t i = 0; i < shard.prefill_graph->inputs.size() && i < shard.prefill_input_holders.size(); ++i) { // [spagetti] tensor_t 만드는 방식이 이런식으로 하는게 맞나?
    const auto& t = shard.prefill_graph->inputs[i];
    
    // Check KV override first
    void* buf = nullptr;
    auto ko = kv_override.find(t.name);
    if (ko != kv_override.end()) {
      buf = ko->second;
    } else {
      auto it = bindings.find(t.name);
      if (it != bindings.end()) buf = it->second;
    }
    
    if (buf) {
      shard.prefill_input_holders[i]->update_buffer(buf, t.nbytes);
      inputs.push_back(shard.prefill_input_holders[i]->tensor());
    }
  }
  
  // Setup outputs (NO KV cache override - we'll copy outputs after execution)
  for (size_t i = 0; i < shard.prefill_graph->outputs.size() && i < shard.prefill_output_holders.size(); ++i) {
    const auto& t = shard.prefill_graph->outputs[i];
    
    auto it = bindings.find(t.name);
    if (it == bindings.end()) continue;
    
    shard.prefill_output_holders[i]->update_buffer(it->second, t.nbytes);
    outputs.push_back(shard.prefill_output_holders[i]->tensor());
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Shard " << shard_idx << "] Executing with " << inputs.size() 
              << " inputs, " << outputs.size() << " outputs...\n";
  }
  
  if (!loader_->execute_graph(shard_idx, "prefill_forward", inputs, outputs)) {
    error_msg_ = "Shard " + std::to_string(shard_idx) + " prefill execution failed";
    return false;
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Shard " << shard_idx << "] Execute completed\n";
  }
  
  // Copy outputs to shared buffers for next shard
  if (config_.log_level >= 2) {
    std::cout << "[Shard " << shard_idx << "] Copying outputs...\n";
    if (shard_idx == 7) {
      std::cout << "[Shard 7] All outputs:\n";
      for (const auto& t : shard.prefill_graph->outputs) {
        std::cout << "  - " << t.name << " : " << t.nbytes << " bytes\n";
      }
    }
  }
  
  for (const auto& t : shard.prefill_graph->outputs) {
    auto it = bindings.find(t.name);
    if (it == bindings.end()) continue;
    
    // ROPE outputs (shard 0 only)
    if (shard_idx == 0) {
      if (t.name.find("output_quantized_decomposed_dequantize_per_tensor_tensor_0") != std::string::npos &&
          t.name.find("_1_0") == std::string::npos) {
        std::memcpy(shared_buffer_views_["rope_cos"], it->second, t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Shard 0] ROPE cos copied: " << t.nbytes << " bytes\n";
        }
      } else if (t.name.find("output_quantized_decomposed_dequantize_per_tensor_tensor_1_0") != std::string::npos) {
        std::memcpy(shared_buffer_views_["rope_sin"], it->second, t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Shard 0] ROPE sin copied: " << t.nbytes << " bytes\n";
        }
      }
    }
    
    // Hidden state output (all shards) - but NOT logits!
    // Hidden state is [1, ar_len, dim] = 131072 bytes, logits are much larger
    if (t.name.find("output_aten_add_tensor") != std::string::npos ||
        t.name.find("fallback") != std::string::npos) {
      std::memcpy(shared_buffer_views_["hidden_state"], it->second, t.nbytes);
      if (config_.log_level >= 2) {
        std::cout << "[Shard " << shard_idx << "] Hidden state copied: " 
                  << t.nbytes << " bytes (" << t.name << ")\n";
        if (shard_idx == 0) {
          // Print first 8 uint16_t values for debugging
          uint16_t* data = (uint16_t*)it->second;
          std::cout << "[Shard 0] Hidden state first 8 values: ";
          for (int i = 0; i < 8; ++i) {
            std::cout << data[i] << " ";
          }
          std::cout << "\n";
        }
      }
    }
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Shard " << shard_idx << "] Output copy completed\n";
  }
  
  if (config_.log_level >= 1) {
    std::cout << "[Shard " << shard_idx << " Prefill] ✓\n";
  }
  
  return true;
}

bool LLMDecodeRunner::run_shard_decode(int shard_idx,
                                        int32_t n_past) {
  if (config_.log_level >= 2) {
    std::cout << "[Decode Shard " << shard_idx << "] n_past=" << n_past << "\n";
  }
  
  auto& shard = shards_[shard_idx];
  auto& bindings = shard.kv_alloc->bindings();
  
  // Setup KV cache INPUT override only
  std::map<std::string, void*> kv_override;
  int shard_layer_start = shard_idx * layers_per_shard_;
  int shard_layer_end = shard_layer_start + layers_per_shard_;
  
  // Input KV cache binding (by dimensions)
  int v_in = 0, k_in = 0;
  for (const auto& t : shard.kv_graph->inputs) {
    if (t.name.find("_args_") != std::string::npos && t.dims.size() == 3) {
      // V cache: [1, cache_len, head_dim=64]
      // K cache: [1, head_dim=64, cache_len]
      bool is_v = (t.dims[2] == head_dim_);
      bool is_k = (t.dims[1] == head_dim_);
      
      if (is_v || is_k) {
        int local_idx = is_v ? v_in++ : k_in++;
        int local_layer = local_idx / num_heads_;
        int head = local_idx % num_heads_;
        int global_layer = shard_layer_start + local_layer;
        
        if (global_layer < num_layers_) {
          const auto& buf = is_v ? kv_manager_->get_v_cache(global_layer, head) : kv_manager_->get_k_cache(global_layer, head);
          kv_override[t.name] = buf.input_buffer;
        }
      }
    }
  }
  
  // Fill inputs (shard 0 already filled in run_multi_context_decode_step)
  if (shard_idx > 0) {
    // Shard 1-7: copy from shared buffers
    for (const auto& t : shard.kv_graph->inputs) {
      if (kv_override.find(t.name) != kv_override.end()) continue;
      auto it = bindings.find(t.name);
      if (it == bindings.end()) continue;
      
      std::string name_lower = t.name;
      for (auto& c : name_lower) c = (char)tolower(c);
      
      if (name_lower.find("fallback") != std::string::npos) {
        std::memcpy(it->second, shared_buffer_views_["hidden_state"], t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Decode Shard " << shard_idx << "] Input hidden state copied: " << t.nbytes << " bytes\n";
        }
      } else if (t.name.find("input_9_aten_view_copy_default_0") != std::string::npos || t.name.find("input_9_aten_select_copy_int_0") != std::string::npos) {
        std::memcpy(it->second, shared_buffer_views_["rope_cos"], t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Decode Shard " << shard_idx << "] ROPE cos copied: " << t.nbytes << " bytes\n";
        }
      } else if (t.name.find("input_10_aten_view_copy_default_1_0") != std::string::npos || t.name.find("input_10_aten_select_copy_int_1_0") != std::string::npos) {
        std::memcpy(it->second, shared_buffer_views_["rope_sin"], t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Decode Shard " << shard_idx << "] ROPE sin copied: " << t.nbytes << " bytes\n";
        }
      } else if (name_lower.find("atten_mask") != std::string::npos) {
        std::memcpy(it->second, shared_buffer_views_["attention_mask"], t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Decode Shard " << shard_idx << "] Attention mask copied: " << t.nbytes << " bytes\n";
        }
      }
    }
  }
  
  // Build tensors and execute
  std::vector<Qnn_Tensor_t> inputs, outputs;
  for (size_t i = 0; i < shard.kv_graph->inputs.size() && i < shard.kv_input_holders.size(); ++i) {
    const auto& t = shard.kv_graph->inputs[i];
    void* buf = nullptr;
    auto ko = kv_override.find(t.name);
    if (ko != kv_override.end()) buf = ko->second;
    else { auto it = bindings.find(t.name); if (it != bindings.end()) buf = it->second; }
    if (buf) { shard.kv_input_holders[i]->update_buffer(buf, t.nbytes); inputs.push_back(shard.kv_input_holders[i]->tensor()); }
  }
  
  for (size_t i = 0; i < shard.kv_graph->outputs.size() && i < shard.kv_output_holders.size(); ++i) {
    const auto& t = shard.kv_graph->outputs[i];
    auto it = bindings.find(t.name);
    if (it == bindings.end()) continue;
    shard.kv_output_holders[i]->update_buffer(it->second, t.nbytes);
    outputs.push_back(shard.kv_output_holders[i]->tensor());
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Decode Shard " << shard_idx << "] Executing with " << inputs.size() << " inputs, " << outputs.size() << " outputs...\n";
  }
  
  if (!loader_->execute_graph(shard_idx, "kv_forward", inputs, outputs)) {
    error_msg_ = "Shard " + std::to_string(shard_idx) + " decode execution failed";
    return false;
  }
  
  if (config_.log_level >= 2) {
    std::cout << "[Decode Shard " << shard_idx << "] Execute completed\n";
  }
  
  // Copy outputs to shared buffers
  for (const auto& t : shard.kv_graph->outputs) {
    auto it = bindings.find(t.name);
    if (it == bindings.end()) continue;
    
    if (shard_idx == 0) {
      if (t.name.find("output_quantized_decomposed_dequantize_per_tensor_tensor_0") != std::string::npos && t.name.find("_1_0") == std::string::npos) {
        std::memcpy(shared_buffer_views_["rope_cos"], it->second, t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Decode Shard 0] ROPE cos copied: " << t.nbytes << " bytes\n";
        }
      } else if (t.name.find("output_quantized_decomposed_dequantize_per_tensor_tensor_1_0") != std::string::npos) {
        std::memcpy(shared_buffer_views_["rope_sin"], it->second, t.nbytes);
        if (config_.log_level >= 2) {
          std::cout << "[Decode Shard 0] ROPE sin copied: " << t.nbytes << " bytes\n";
        }
      }
    }
    
    // Hidden state output (NOT logits!) - size check
    if (t.name.find("output_aten_add_tensor") != std::string::npos || 
        t.name.find("fallback") != std::string::npos) {
      std::memcpy(shared_buffer_views_["hidden_state"], it->second, t.nbytes);
      if (config_.log_level >= 2) {
        std::cout << "[Decode Shard " << shard_idx << "] Hidden state copied: " << t.nbytes << " bytes\n";
      }
    }
  }
  
  return true;
}

int LLMDecodeRunner::qnn_decode(llama_context * ctx, llama_batch batch) {
  if (batch.n_tokens <= 0) {
    error_msg_ = "Empty batch";
    return -1;
  }
  
  bool is_prefill = (batch.n_tokens > 1);
  
  if (is_prefill) {
    // Prefill: process all tokens (n_past_ updated inside)
    if (config_.use_multi_context) {
      if (!run_multi_context_prefill(ctx, batch)) {
        return -1;
      }
    } else {
      // Single-context fallback (legacy interface)
      std::vector<int32_t> tokens(batch.n_tokens);
      for (int i = 0; i < batch.n_tokens; ++i) {
        tokens[i] = batch.token[i];
      }
      int32_t next_token = 0;
      int32_t n_update = 0;
      if (!run_prefill(tokens, next_token, n_update, ctx)) {
        return -1;
      }
      n_past_ = n_update;
    }
  } else {
    // Decode: single token step (n_past_ updated inside)
    if (config_.use_multi_context) {
      if (!run_multi_context_decode_step(ctx, batch)) {
        return -1;
      }
    } else {
      // Single-context fallback (legacy interface)
      int32_t token_out = 0;
      int32_t token_in = batch.token[0];
      if (!run_decode_step(token_in, n_past_, token_out, ctx)) {
        return -1;
      }
      n_past_ += 1;
    }
  }
  
  return 0;  // Success (matches llama_decode return semantics)
}

} // namespace llama_qnn
