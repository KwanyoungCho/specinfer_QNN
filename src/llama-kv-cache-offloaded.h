#pragma once

#include "llama-kv-cache.h"
#include "ggml-backend.h"

#include <thread>
#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <string>
#include <cstdint>

// Forward declaration
struct llama_model;

//
// llama_kv_cache_offloaded
//
// Extends llama_kv_cache with SSD offloading using a fixed-size memory pool.
// Only m slots are allocated in memory (m << L), with the rest offloaded to SSD.
// Uses cb_eval callback for layer-level prefetch/save orchestration.
//

class llama_kv_cache_offloaded : public llama_kv_cache {
public:
    llama_kv_cache_offloaded(
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
            const std::string & cache_dir);

    ~llama_kv_cache_offloaded();

    // =========================================================================
    // Slot & pool info
    // =========================================================================

    uint32_t get_pool_slots()   const { return m_pool_slots; }
    uint32_t get_total_layers() const { return m_total_layers; }

    // =========================================================================
    // Prefetch / save orchestration (called from cb_eval or externally)
    // =========================================================================

    // Pre-warm pool during draft phase: async load layers 0..m-2 into host buffers
    void prepare_target_pass();

    // Wait for a specific layer's prefetch to complete, then tensor_set from host buffer
    // Must be called from main thread (backend synchronized)
    void wait_and_load_layer(uint32_t model_layer);

    // Save delta for a layer after its compute completes
    // Reads from tensor to host buffer, then async writes to SSD
    void save_layer_delta(uint32_t model_layer, uint32_t cell_start, uint32_t cell_count);

    // Notify that target verification is complete with accepted token count
    void on_verify_complete(uint32_t n_accepted);

    // Queue an async prefetch: SSD -> host buffer for a given model layer
    void queue_prefetch(uint32_t model_layer);

    // Check if a layer's prefetch is complete (non-blocking)
    bool is_prefetch_ready(uint32_t model_layer) const;

    // =========================================================================
    // cb_eval callback (static, to be registered with ggml_backend_sched)
    // =========================================================================

    // The combined callback that handles both hidden state capture and KV offloading
    static bool eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data);

    // =========================================================================
    // Stats
    // =========================================================================

    struct stats {
        uint32_t prefetch_hits   = 0;
        uint32_t prefetch_stalls = 0;
        uint32_t saves           = 0;
        uint32_t loads           = 0;
    };

    stats get_stats() const { return m_stats; }
    void  reset_stats() { m_stats = {}; }

private:
    // =========================================================================
    // Configuration
    // =========================================================================

    uint32_t    m_pool_slots;     // m: number of memory pool slots
    uint32_t    m_total_layers;   // L: total model layers with KV cache
    std::string m_cache_dir;

    // KV size info (cached from hparams for convenience)
    uint32_t m_kv_size;       // number of KV cells (sequence length budget)
    ggml_type m_type_k;
    ggml_type m_type_v;
    bool      m_v_trans;

    // =========================================================================
    // Per-layer SSD storage
    // =========================================================================

    struct ssd_layer {
        int      fd_k = -1;    // file descriptor for K data
        int      fd_v = -1;    // file descriptor for V data
        size_t   k_total_bytes = 0;
        size_t   v_total_bytes = 0;
    };

    std::vector<ssd_layer> m_ssd_layers; // [L]

    void open_ssd_files();
    void close_ssd_files();

    // =========================================================================
    // Host buffers (one per pool slot, for 2-stage I/O)
    // =========================================================================

    struct host_buffer {
        void *  k_data = nullptr;
        void *  v_data = nullptr;
        size_t  k_size = 0;
        size_t  v_size = 0;
    };

    std::vector<host_buffer> m_host_bufs; // [m]

    void alloc_host_buffers();
    void free_host_buffers();

    // =========================================================================
    // Slot state tracking
    // =========================================================================

    struct slot_state {
        int32_t             resident_layer = -1;  // which model layer's data is currently in this slot (-1 = empty)
        std::atomic<bool>   prefetch_pending{false};
        std::atomic<bool>   prefetch_complete{false};
        int32_t             prefetch_target_layer = -1; // which layer is being prefetched
        std::atomic<bool>   save_pending{false};  // true while host_buf is being read by worker (SSD write in progress)
        std::mutex          mtx;
        std::condition_variable cv;
    };

    std::deque<slot_state> m_slots; // [m]  (deque: no realloc, supports non-movable types)

    uint32_t layer_to_slot(uint32_t model_layer) const { return model_layer % m_pool_slots; }

    // =========================================================================
    // Async I/O engine
    // =========================================================================

    enum class io_op { LOAD, SAVE };

    struct io_task {
        io_op    op;
        uint32_t model_layer;
        uint32_t slot_id;
        // For delta save
        uint32_t cell_start;
        uint32_t cell_count;
    };

    std::thread              m_io_thread;
    std::queue<io_task>      m_io_queue;
    std::mutex               m_io_mutex;
    std::condition_variable  m_io_cv;
    std::atomic<bool>        m_shutdown{false};

    void io_worker_loop();

    // I/O operations (run on worker thread)
    void execute_load(const io_task & task);
    void execute_save(const io_task & task);

    // Synchronous tensor read/write (must be called from main thread when backend is sync'd)
    void tensor_to_host(uint32_t model_layer, uint32_t slot_id);
    void tensor_to_host_delta(uint32_t model_layer, uint32_t slot_id, uint32_t cell_start, uint32_t cell_count);
    void host_to_tensor(uint32_t model_layer, uint32_t slot_id);

    // SSD file I/O (can be called from any thread)
    void ssd_read_layer(uint32_t model_layer, uint32_t slot_id);
    void ssd_write_delta(uint32_t model_layer, uint32_t slot_id, uint32_t cell_start, uint32_t cell_count);

    // =========================================================================
    // Callback state (for compose with existing callbacks)
    // =========================================================================

    // User's original callback (e.g., cb_get_hidden)
    ggml_backend_sched_eval_callback m_user_cb      = nullptr;
    void *                           m_user_cb_data = nullptr;

    // Whether we're in the initial prefill pass (no prior data on SSD)
    bool m_is_prefill = true;

    // Track which layer we last completed in the current graph compute
    int32_t m_last_completed_layer = -1;

    // Delta info for current ubatch (set before graph compute, used in eval_callback)
    uint32_t m_delta_head  = 0;  // cell index where new tokens start
    uint32_t m_delta_count = 0;  // number of new tokens written

    // =========================================================================
    // Stats
    // =========================================================================

    stats m_stats;

public:
    // Set user callback to compose with
    void set_user_callback(ggml_backend_sched_eval_callback cb, void * data) {
        m_user_cb      = cb;
        m_user_cb_data = data;
    }

    void set_prefill(bool is_prefill) { m_is_prefill = is_prefill; }
    void reset_layer_tracking() { m_last_completed_layer = -1; }

    // Set delta info before each graph compute (called from apply())
    void set_delta_info(uint32_t head, uint32_t count) {
        m_delta_head  = head;
        m_delta_count = count;
    }
};

// Callback data struct that combines offloader + user callback data
struct kv_offload_cb_data {
    llama_kv_cache_offloaded * offloader = nullptr;
    // User callback data (e.g., callback_data for hidden state capture)
    void * user_data = nullptr;
};
