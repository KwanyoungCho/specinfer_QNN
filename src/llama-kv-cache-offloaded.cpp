/*
 * ============================================================================
 * KV Cache SSD Offloading with Memory Pool
 * ============================================================================
 *
 * Extends llama_kv_cache with a fixed-size memory pool (m slots) and SSD
 * offloading. Uses cb_eval callback for layer-level prefetch/save control.
 *
 * I/O Pipeline (2-stage for thread safety):
 *   Worker thread: SSD <-> host buffer
 *   Main thread (cb_eval): host buffer <-> ggml tensor
 */

#include "llama-kv-cache-offloaded.h"
#include "llama-hparams.h"
#include "llama-model.h"

#include <cstring>
#include <cstdio>
#include <cassert>
#include <filesystem>

// POSIX I/O for pread/pwrite
#include <fcntl.h>
#include <unistd.h>

// Set to 1 to enable verbose KV offload tracing
#define KV_OFFLOAD_DEBUG 0
#if KV_OFFLOAD_DEBUG
#define KV_DBG(...) fprintf(stderr, __VA_ARGS__)
#else
#define KV_DBG(...) ((void)0)
#endif
#include <sys/stat.h>

// ============================================================================
// Construction / Destruction
// ============================================================================

static llama_kv_cache::layer_filter_cb make_pool_filter(uint32_t n_pool_slots) {
    return [n_pool_slots](uint32_t il) -> bool {
        return il < n_pool_slots;
    };
}

static llama_kv_cache::layer_reuse_cb make_pool_reuse(uint32_t n_pool_slots) {
    return [n_pool_slots](int32_t il) -> int32_t {
        if ((uint32_t)il < n_pool_slots) return -1; // first m layers own their tensors
        return (int32_t)((uint32_t)il % n_pool_slots);
    };
}

llama_kv_cache_offloaded::llama_kv_cache_offloaded(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                 uint32_t   n_pool_slots,
        const std::string & cache_dir)
    : llama_kv_cache(model, type_k, type_v, v_trans, offload, unified,
                     kv_size, n_seq_max, n_pad, n_swa, swa_type,
                     make_pool_filter(n_pool_slots),
                     make_pool_reuse(n_pool_slots)),
      m_pool_slots(n_pool_slots),
      m_total_layers(model.hparams.n_layer),
      m_cache_dir(cache_dir),
      m_kv_size(kv_size),
      m_type_k(type_k),
      m_type_v(type_v),
      m_v_trans(v_trans)
{
    // Clamp pool slots to total layers
    if (m_pool_slots > m_total_layers) {
        m_pool_slots = m_total_layers;
    }

    printf("[KV-Offload] Initializing: %u pool slots, %u total layers, kv_size=%u\n",
           m_pool_slots, m_total_layers, m_kv_size);
    printf("[KV-Offload] Cache dir: %s\n", m_cache_dir.c_str());

    // Initialize SSD layer descriptors
    m_ssd_layers.resize(m_total_layers);
    for (uint32_t il = 0; il < m_total_layers; il++) {
        const auto & hp = get_hparams();
        uint32_t n_embd_k = hp.n_embd_k_gqa(il);
        uint32_t n_embd_v = m_v_trans ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);

        m_ssd_layers[il].k_total_bytes = (size_t)n_embd_k * m_kv_size * ggml_type_size(m_type_k) / ggml_blck_size(m_type_k);
        m_ssd_layers[il].v_total_bytes = (size_t)n_embd_v * m_kv_size * ggml_type_size(m_type_v) / ggml_blck_size(m_type_v);
    }

    // Initialize slot states.
    // slot_state contains non-movable types (std::mutex, std::condition_variable,
    // std::atomic), so we use std::deque which never relocates existing elements.
    for (uint32_t i = 0; i < m_pool_slots; i++) {
        m_slots.emplace_back();
    }

    // Allocate host buffers (LOAD-only: SSD -> host -> GPU)
    alloc_host_buffers();

    // Allocate staging buffers (SAVE-only: GPU -> staging -> SSD)
    alloc_staging_buffers();

    // Open SSD files
    open_ssd_files();

    // Start I/O worker thread
    m_io_thread = std::thread(&llama_kv_cache_offloaded::io_worker_loop, this);

    printf("[KV-Offload] Ready. Per-layer K size: %.2f MB, V size: %.2f MB\n",
           m_ssd_layers[0].k_total_bytes / (1024.0 * 1024.0),
           m_ssd_layers[0].v_total_bytes / (1024.0 * 1024.0));
}

llama_kv_cache_offloaded::~llama_kv_cache_offloaded() {
    m_shutdown.store(true);
    m_io_cv.notify_all();

    if (m_io_thread.joinable()) {
        m_io_thread.join();
    }

    close_ssd_files();
    free_staging_buffers();
    free_host_buffers();
}

// ============================================================================
// Host Buffer Management
// ============================================================================

void llama_kv_cache_offloaded::alloc_host_buffers() {
    m_host_bufs.resize(m_pool_slots);

    for (uint32_t s = 0; s < m_pool_slots; s++) {
        // Use the max layer sizes (layer 0 is representative for uniform models)
        size_t k_sz = m_ssd_layers[0].k_total_bytes;
        size_t v_sz = m_ssd_layers[0].v_total_bytes;

        m_host_bufs[s].k_data = malloc(k_sz);
        m_host_bufs[s].v_data = malloc(v_sz);
        m_host_bufs[s].k_size = k_sz;
        m_host_bufs[s].v_size = v_sz;

        if (!m_host_bufs[s].k_data || !m_host_bufs[s].v_data) {
            throw std::runtime_error("[KV-Offload] Failed to allocate host buffers");
        }

        memset(m_host_bufs[s].k_data, 0, k_sz);
        memset(m_host_bufs[s].v_data, 0, v_sz);
    }

    printf("[KV-Offload] Allocated %u host buffers (%.2f MB each)\n",
           m_pool_slots,
           (m_host_bufs[0].k_size + m_host_bufs[0].v_size) / (1024.0 * 1024.0));
}

void llama_kv_cache_offloaded::free_host_buffers() {
    for (auto & buf : m_host_bufs) {
        free(buf.k_data); buf.k_data = nullptr;
        free(buf.v_data); buf.v_data = nullptr;
    }
}

// ============================================================================
// Staging Buffer Management (SAVE-only: GPU → staging → SSD)
// ============================================================================

void llama_kv_cache_offloaded::alloc_staging_buffers() {
    m_staging_bufs.resize(m_total_layers);

    const auto & hp = get_hparams();

    for (uint32_t il = 0; il < m_total_layers; il++) {
        uint32_t n_embd_k = hp.n_embd_k_gqa(il);
        uint32_t n_embd_v = hp.n_embd_v_gqa(il);  // non-transposed only

        size_t k_row = (size_t)n_embd_k * ggml_type_size(m_type_k) / ggml_blck_size(m_type_k);
        size_t v_row = (size_t)n_embd_v * ggml_type_size(m_type_v) / ggml_blck_size(m_type_v);

        size_t k_cap = k_row * MAX_STAGING_TOKENS;
        size_t v_cap = v_row * MAX_STAGING_TOKENS;

        m_staging_bufs[il].k_data     = malloc(k_cap);
        m_staging_bufs[il].v_data     = malloc(v_cap);
        m_staging_bufs[il].k_capacity = k_cap;
        m_staging_bufs[il].v_capacity = v_cap;

        if (!m_staging_bufs[il].k_data || !m_staging_bufs[il].v_data) {
            throw std::runtime_error("[KV-Offload] Failed to allocate staging buffers");
        }
    }

    size_t total_staging = 0;
    for (auto & sb : m_staging_bufs) total_staging += sb.k_capacity + sb.v_capacity;
    printf("[KV-Offload] Allocated %u staging buffers (%.2f MB total, max %u tokens)\n",
           m_total_layers, total_staging / (1024.0 * 1024.0), MAX_STAGING_TOKENS);
}

void llama_kv_cache_offloaded::free_staging_buffers() {
    for (auto & sb : m_staging_bufs) {
        free(sb.k_data); sb.k_data = nullptr;
        free(sb.v_data); sb.v_data = nullptr;
    }
}

// ============================================================================
// SSD File Management
// ============================================================================

void llama_kv_cache_offloaded::open_ssd_files() {
    std::filesystem::create_directories(m_cache_dir);

    for (uint32_t il = 0; il < m_total_layers; il++) {
        std::string k_path = m_cache_dir + "/layer_" + std::to_string(il) + "_K.bin";
        std::string v_path = m_cache_dir + "/layer_" + std::to_string(il) + "_V.bin";

        m_ssd_layers[il].fd_k = open(k_path.c_str(), O_RDWR | O_CREAT, 0644);
        m_ssd_layers[il].fd_v = open(v_path.c_str(), O_RDWR | O_CREAT, 0644);

        if (m_ssd_layers[il].fd_k < 0 || m_ssd_layers[il].fd_v < 0) {
            throw std::runtime_error("[KV-Offload] Failed to open SSD files for layer " + std::to_string(il));
        }

        // Pre-allocate file to full size
        ftruncate(m_ssd_layers[il].fd_k, (off_t)m_ssd_layers[il].k_total_bytes);
        ftruncate(m_ssd_layers[il].fd_v, (off_t)m_ssd_layers[il].v_total_bytes);
    }
}

void llama_kv_cache_offloaded::close_ssd_files() {
    for (auto & sl : m_ssd_layers) {
        if (sl.fd_k >= 0) { close(sl.fd_k); sl.fd_k = -1; }
        if (sl.fd_v >= 0) { close(sl.fd_v); sl.fd_v = -1; }
    }
}

// ============================================================================
// Async I/O Engine
// ============================================================================

void llama_kv_cache_offloaded::io_worker_loop() {
    while (!m_shutdown.load()) {
        io_task task;

        {
            std::unique_lock<std::mutex> lock(m_io_mutex);
            m_io_cv.wait(lock, [this] {
                return !m_io_queue.empty() || m_shutdown.load();
            });

            if (m_shutdown.load() && m_io_queue.empty()) break;
            if (m_io_queue.empty()) continue;

            task = m_io_queue.front();
            m_io_queue.pop();
        }

        execute_load(task);
    }
}

void llama_kv_cache_offloaded::execute_load(const io_task & task) {
    KV_DBG("[KV-DBG] worker: LOAD layer=%u slot=%u\n", task.model_layer, task.slot_id);
    ssd_read_layer(task.model_layer, task.slot_id);

    auto & slot = m_slots[task.slot_id];
    {
        std::lock_guard<std::mutex> lock(slot.mtx);
        slot.prefetch_complete.store(true);
        slot.prefetch_pending.store(false);
        slot.prefetch_target_layer = (int32_t)task.model_layer;
    }
    slot.cv.notify_all();
    m_stats.loads++;
}

// ============================================================================
// SSD Read/Write
// ============================================================================

void llama_kv_cache_offloaded::ssd_read_layer(uint32_t model_layer, uint32_t slot_id) {
    auto & sl  = m_ssd_layers[model_layer];
    auto & buf = m_host_bufs[slot_id];

    ssize_t kr = pread(sl.fd_k, buf.k_data, sl.k_total_bytes, 0);
    ssize_t vr = pread(sl.fd_v, buf.v_data, sl.v_total_bytes, 0);

    if (kr < (ssize_t)sl.k_total_bytes || vr < (ssize_t)sl.v_total_bytes) {
        fprintf(stderr, "[KV-Offload] WARNING: pread layer %u: k=%zd/%zu v=%zd/%zu (short read!)\n",
                model_layer, kr, sl.k_total_bytes, vr, sl.v_total_bytes);
    }
}

void llama_kv_cache_offloaded::ssd_write_full_layer(uint32_t model_layer, uint32_t slot_id) {
    auto & sl  = m_ssd_layers[model_layer];
    auto & buf = m_host_bufs[slot_id];

    ssize_t kw = pwrite(sl.fd_k, buf.k_data, sl.k_total_bytes, 0);
    ssize_t vw = pwrite(sl.fd_v, buf.v_data, sl.v_total_bytes, 0);

    if (kw < (ssize_t)sl.k_total_bytes || vw < (ssize_t)sl.v_total_bytes) {
        fprintf(stderr, "[KV-Offload] WARNING: pwrite full layer %u: k=%zd/%zu v=%zd/%zu\n",
                model_layer, kw, sl.k_total_bytes, vw, sl.v_total_bytes);
    }
}

void llama_kv_cache_offloaded::ssd_write_staging_delta(uint32_t model_layer) {
    if (m_staging_delta_count == 0) return;

    auto & sl  = m_ssd_layers[model_layer];
    auto & stg = m_staging_bufs[model_layer];

    const auto & hp = get_hparams();
    uint32_t n_embd_k = hp.n_embd_k_gqa(model_layer);

    // K: staging stores compact [0, delta_count * k_row), write at SSD offset
    {
        size_t k_row = (size_t)n_embd_k * ggml_type_size(m_type_k) / ggml_blck_size(m_type_k);
        size_t k_off = (size_t)m_staging_delta_head * k_row;
        size_t k_len = (size_t)m_staging_delta_count * k_row;

        ssize_t w = pwrite(sl.fd_k, stg.k_data, k_len, (off_t)k_off);
        if (w < 0) {
            printf("[KV-Offload] WARNING: pwrite staging K failed for layer %u\n", model_layer);
        }
    }

    // V: only when !v_trans (cells contiguous).  When v_trans, V is saved
    // directly in eval_callback via save_v_full() (full tensor → SSD).
    if (!m_v_trans) {
        uint32_t n_embd_v = hp.n_embd_v_gqa(model_layer);
        size_t v_row = (size_t)n_embd_v * ggml_type_size(m_type_v) / ggml_blck_size(m_type_v);
        size_t v_off = (size_t)m_staging_delta_head * v_row;
        size_t v_len = (size_t)m_staging_delta_count * v_row;

        ssize_t w = pwrite(sl.fd_v, stg.v_data, v_len, (off_t)v_off);
        if (w < 0) {
            printf("[KV-Offload] WARNING: pwrite staging V failed for layer %u\n", model_layer);
        }
    }
}

// ============================================================================
// Tensor <-> Host Buffer (main thread only, backend must be synchronized)
// ============================================================================

void llama_kv_cache_offloaded::tensor_to_host(uint32_t model_layer, uint32_t slot_id) {
    ggml_tensor * k = get_layer_k(slot_id);
    ggml_tensor * v = get_layer_v(slot_id);
    auto & buf = m_host_bufs[slot_id];
    auto & sl  = m_ssd_layers[model_layer];

    ggml_backend_tensor_get(k, buf.k_data, 0, sl.k_total_bytes);
    ggml_backend_tensor_get(v, buf.v_data, 0, sl.v_total_bytes);
}

void llama_kv_cache_offloaded::capture_delta_to_staging(uint32_t model_layer, uint32_t slot_id) {
    uint32_t cell_start = m_delta_head;
    uint32_t cell_count = m_delta_count;
    if (cell_count == 0) return;

    assert(cell_count <= MAX_STAGING_TOKENS);

    ggml_tensor * k = get_layer_k(slot_id);
    auto & stg = m_staging_bufs[model_layer];

    const auto & hp = get_hparams();
    uint32_t n_embd_k = hp.n_embd_k_gqa(model_layer);

    // K: read from GPU at [cell_start, cell_start+cell_count) → staging compact [0, k_len)
    {
        size_t k_row = (size_t)n_embd_k * ggml_type_size(m_type_k) / ggml_blck_size(m_type_k);
        size_t k_off = (size_t)cell_start * k_row;
        size_t k_len = (size_t)cell_count * k_row;
        ggml_backend_tensor_get(k, stg.k_data, k_off, k_len);
    }

    // V: only when !v_trans (cells contiguous).  When v_trans, V is saved
    // directly from GPU → host_buf → SSD via save_v_full() in eval_callback.
    if (!m_v_trans) {
        ggml_tensor * v = get_layer_v(slot_id);
        uint32_t n_embd_v = hp.n_embd_v_gqa(model_layer);
        size_t v_row = (size_t)n_embd_v * ggml_type_size(m_type_v) / ggml_blck_size(m_type_v);
        size_t v_off = (size_t)cell_start * v_row;
        size_t v_len = (size_t)cell_count * v_row;
        ggml_backend_tensor_get(v, stg.v_data, v_off, v_len);
    }
}

void llama_kv_cache_offloaded::host_to_tensor(uint32_t model_layer, uint32_t slot_id) {
    ggml_tensor * k = get_layer_k(slot_id);
    ggml_tensor * v = get_layer_v(slot_id);
    auto & buf = m_host_bufs[slot_id];
    auto & sl  = m_ssd_layers[model_layer];

    ggml_backend_tensor_set(k, buf.k_data, 0, sl.k_total_bytes);
    ggml_backend_tensor_set(v, buf.v_data, 0, sl.v_total_bytes);
}

// Full V save for v_trans mode: GPU → host_buf → SSD (synchronous).
// When v_trans=true the V cells are scattered (column-major), so the
// compact staging delta path cannot be used.  Instead we copy the entire
// V tensor through the host buffer each time a layer completes.
void llama_kv_cache_offloaded::save_v_full(uint32_t model_layer, uint32_t slot_id) {
    ggml_tensor * v = get_layer_v(slot_id);
    auto & buf = m_host_bufs[slot_id];
    auto & sl  = m_ssd_layers[model_layer];

    ggml_backend_tensor_get(v, buf.v_data, 0, sl.v_total_bytes);

    ssize_t w = pwrite(sl.fd_v, buf.v_data, sl.v_total_bytes, 0);
    if (w < (ssize_t)sl.v_total_bytes) {
        fprintf(stderr, "[KV-Offload] WARNING: save_v_full layer %u: %zd/%zu\n",
                model_layer, w, sl.v_total_bytes);
    }
}

// ============================================================================
// Prefetch / Save Orchestration
// ============================================================================

void llama_kv_cache_offloaded::queue_prefetch(uint32_t model_layer) {
    if (model_layer >= m_total_layers) return;

    uint32_t slot = layer_to_slot(model_layer);
    auto & s = m_slots[slot];

    if (s.prefetch_pending.load() && s.prefetch_target_layer == (int32_t)model_layer) return;
    if (s.prefetch_complete.load() && s.prefetch_target_layer == (int32_t)model_layer) return;

    {
        std::lock_guard<std::mutex> lock(s.mtx);
        s.prefetch_pending.store(true);
        s.prefetch_complete.store(false);
        s.prefetch_target_layer = (int32_t)model_layer;
    }

    io_task task;
    task.model_layer = model_layer;
    task.slot_id     = slot;

    {
        std::lock_guard<std::mutex> lock(m_io_mutex);
        m_io_queue.push(task);
    }
    m_io_cv.notify_one();
}

bool llama_kv_cache_offloaded::is_prefetch_ready(uint32_t model_layer) const {
    uint32_t slot = layer_to_slot(model_layer);
    auto & s = m_slots[slot];
    return s.prefetch_complete.load() && s.prefetch_target_layer == (int32_t)model_layer;
}

void llama_kv_cache_offloaded::wait_and_load_layer(uint32_t model_layer) {
    uint32_t slot = layer_to_slot(model_layer);
    auto & s = m_slots[slot];

    if (s.resident_layer == (int32_t)model_layer && !s.prefetch_pending.load()) {
        m_stats.prefetch_hits++;
        KV_DBG("[KV-DBG] wait_and_load: layer=%u slot=%u HIT (resident)\n", model_layer, slot);
        return;
    }

    if (s.prefetch_pending.load() || !s.prefetch_complete.load()) {
        KV_DBG("[KV-DBG] wait_and_load: layer=%u slot=%u WAITING\n", model_layer, slot);
        std::unique_lock<std::mutex> lock(s.mtx);
        s.cv.wait(lock, [&] {
            return s.prefetch_complete.load() || m_shutdown.load();
        });
        m_stats.prefetch_stalls++;
    } else {
        m_stats.prefetch_hits++;
    }

    if (m_shutdown.load()) return;

    KV_DBG("[KV-DBG] wait_and_load: layer=%u slot=%u -> host_to_tensor\n", model_layer, slot);
    host_to_tensor(model_layer, slot);

    s.resident_layer = (int32_t)model_layer;
    s.prefetch_complete.store(false);
}

void llama_kv_cache_offloaded::flush_saves() {
    if (!m_staging_dirty) return;

    KV_DBG("[KV-DBG] flush_saves: delta=[%u+%u] for %u layers\n",
           m_staging_delta_head, m_staging_delta_count, m_total_layers);

    for (uint32_t il = 0; il < m_total_layers; il++) {
        ssd_write_staging_delta(il);
    }
    m_staging_dirty = false;
    m_stats.saves += m_total_layers;
}

void llama_kv_cache_offloaded::prepare_target_pass() {
    reset_layer_tracking();

    if (m_is_prefill) {
        KV_DBG("[KV-DBG] prepare_target_pass: PREFILL mode, skipping\n");
        return;
    }

    // 1. Flush staging deltas to SSD (synchronous, tiny)
    flush_saves();

    // 2. Queue async prefetch for layers 0..m-1 (all pool slots)
    uint32_t n_prefetch = std::min(m_pool_slots, m_total_layers);

    KV_DBG("[KV-DBG] prepare_target_pass: flush done, queuing prefetch for layers 0..%u\n",
           n_prefetch - 1);

    for (uint32_t il = 0; il < n_prefetch; il++) {
        queue_prefetch(il);
    }
    // Returns immediately — I/O proceeds in parallel with draft phase.
}

void llama_kv_cache_offloaded::begin_target_verify() {
    if (m_is_prefill) return;

    // 1. Wait for layers 0..m-2 and load into GPU tensors
    uint32_t n_preload = (m_pool_slots > 1) ? (m_pool_slots - 1) : 1;
    if (n_preload > m_total_layers) n_preload = m_total_layers;

    KV_DBG("[KV-DBG] begin_target_verify: wait+load layers 0..%u\n", n_preload - 1);

    for (uint32_t il = 0; il < n_preload; il++) {
        wait_and_load_layer(il);
    }

    // 2. Queue prefetch for layer m-1 so it loads while GPU computes 0..m-2
    if (n_preload < m_total_layers) {
        queue_prefetch(n_preload);
    }
}

void llama_kv_cache_offloaded::on_verify_complete(uint32_t n_accepted) {
    m_is_prefill = false;
    (void)n_accepted;
}

// ============================================================================
// cb_eval Callback Implementation
// ============================================================================

bool llama_kv_cache_offloaded::eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * cb_data = static_cast<kv_offload_cb_data *>(user_data);
    auto * off = cb_data->offloader;

    int il = -1;
    bool is_kqv_out = (sscanf(tensor->name, "kqv_out-%d", &il) == 1);
    bool is_result_norm = (strcmp(tensor->name, "result_norm") == 0);

    if (ask) {
        if (is_kqv_out) return true;
        if (off->m_user_cb) {
            return off->m_user_cb(tensor, true, cb_data->user_data);
        }
        return false;
    }

    // ask == false: node just finished computing

    if (is_result_norm && off->m_user_cb) {
        off->m_user_cb(tensor, false, cb_data->user_data);
        return true;
    }

    if (!is_kqv_out || il < 0) return true;

    // =========================================================================
    // Layer il just completed
    // =========================================================================

    off->m_last_completed_layer = il;
    uint32_t slot = off->layer_to_slot((uint32_t)il);

    KV_DBG("[KV-DBG] eval_cb: layer=%d slot=%u prefill=%d delta=[%u+%u]\n",
           il, slot, off->m_is_prefill, off->m_delta_head, off->m_delta_count);

    if (!off->m_is_prefill) {
        // -----------------------------------------------------------------
        // DECODE path: capture delta to staging (tiny), queue next prefetch
        // -----------------------------------------------------------------

        // ① Capture K delta to staging (V too when !v_trans)
        off->capture_delta_to_staging((uint32_t)il, slot);
        off->m_staging_dirty = true;
        off->m_staging_delta_head  = off->m_delta_head;
        off->m_staging_delta_count = off->m_delta_count;

        // ①b v_trans: V cells are scattered (column-major) so staging can't
        //    handle them.  Save the full V tensor synchronously instead.
        //    host_buf[slot].v_data is used as scratch — queue_prefetch below
        //    will overwrite it with the next layer's data anyway.
        if (off->m_v_trans) {
            off->save_v_full((uint32_t)il, slot);
        }

        // ② Queue prefetch for next layer that will reuse this slot
        uint32_t future_il = (uint32_t)il + off->m_pool_slots;
        if (future_il < off->m_total_layers) {
            off->queue_prefetch(future_il);
        }

        // ③ Wait and load next layer into GPU tensor (if needed)
        uint32_t next_layer = (uint32_t)il + 1;
        if (next_layer < off->m_total_layers) {
            off->wait_and_load_layer(next_layer);
        }
    } else {
        // -----------------------------------------------------------------
        // PREFILL path: synchronous full save (GPU → host_buf → SSD)
        // No loading needed — all layers computed fresh.
        // -----------------------------------------------------------------
        off->tensor_to_host((uint32_t)il, slot);
        off->ssd_write_full_layer((uint32_t)il, slot);
        off->m_stats.saves++;
    }

    return true;
}
