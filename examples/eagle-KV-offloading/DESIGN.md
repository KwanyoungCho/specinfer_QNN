# EAGLE Speculative Decoding + KV Cache SSD Offloading — 설계 및 구현 상세 문서

## 목차

1. [개요 및 핵심 아이디어](#1-개요-및-핵심-아이디어)
2. [아키텍처 전체 흐름도](#2-아키텍처-전체-흐름도)
3. [Memory Pool 설계 (m-slot Circular Buffer)](#3-memory-pool-설계)
4. [3-Tier 데이터 파이프라인](#4-3-tier-데이터-파이프라인)
5. [클래스 구조 — llama_kv_cache_offloaded](#5-클래스-구조)
6. [API 상세 설명](#6-api-상세-설명)
7. [eval_callback 동작 흐름](#7-eval_callback-동작-흐름)
8. [애플리케이션 흐름 — eagle-KV-offloading.cpp](#8-애플리케이션-흐름)
9. [Tree Decoding 로직 (Budget-7 / Budget-25)](#9-tree-decoding-로직)
10. [Delta Save 최적화](#10-delta-save-최적화)
11. [동기화 메커니즘](#11-동기화-메커니즘)
12. [알려진 이슈 및 주의사항](#12-알려진-이슈-및-주의사항)

---

## 1. 개요 및 핵심 아이디어

### 문제
EAGLE 기반 Speculative Decoding에서 **target 모델**은 L개의 transformer layer를 가지며, 각 layer는 자체 KV cache를 GPU 메모리에 유지해야 한다. L이 크면(예: 32 layers) KV cache만으로도 수 GB의 GPU 메모리가 필요하다.

### 해결책
**m-slot Memory Pool + SSD Offloading**: GPU에는 m개의 물리적 KV cache slot만 할당하고(m << L), 나머지 layer의 데이터는 SSD에 저장한다. Target 모델의 각 layer가 실행될 때, 해당 layer의 KV 데이터를 SSD에서 GPU로 적시에 로딩(prefetch)하고, 실행 완료 후 업데이트된 데이터를 SSD에 다시 저장(save)한다.

### 핵심 구성요소

```
┌─────────────────────────────────────────────────┐
│              eagle-KV-offloading.cpp             │
│       (EAGLE Speculative Decoding 메인 루프)      │
│                                                   │
│  [Prefill] → [Tree Drafting] → [Target Verify]  │
│      ↕              ↕               ↕             │
│  offloader API calls for each phase              │
└─────────────┬───────────────────────┬────────────┘
              │                       │
┌─────────────▼───────────────────────▼────────────┐
│         llama_kv_cache_offloaded (클래스)          │
│                                                   │
│  ┌──────────┐  ┌────────────┐  ┌───────────────┐ │
│  │ GPU Pool │  │ Host Bufs  │  │  SSD Files    │ │
│  │ (m slots)│  │ (m buffers)│  │ (L×2 files)   │ │
│  │ K,V 텐서 │←→│ k_data     │←→│ layer_X_K.bin │ │
│  │          │  │ v_data     │  │ layer_X_V.bin │ │
│  └──────────┘  └────────────┘  └───────────────┘ │
│        ↑ eval_callback가 layer별 save/load 제어   │
│        │                                          │
│  ┌─────┴───────────────────────────────────────┐  │
│  │  I/O Worker Thread (비동기 SSD ↔ Host)       │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

### 파일 구조

| 파일 | 역할 |
|------|------|
| `examples/eagle-KV-offloading/eagle-KV-offloading.cpp` | 메인 애플리케이션. EAGLE speculative decoding + offloading 통합 |
| `src/llama-kv-cache-offloaded.h` | `llama_kv_cache_offloaded` 클래스 선언 + `kv_offload_cb_data` 구조체 |
| `src/llama-kv-cache-offloaded.cpp` | 클래스 구현: pool 관리, SSD I/O, eval_callback |
| `src/llama-kv-cache.cpp` | `apply()` 함수에서 delta info를 offloader에 전파 (수정됨) |

---

## 2. 아키텍처 전체 흐름도

### Speculative Decoding 한 스텝의 전체 흐름

```
 ┌─────────────── Speculation Loop (매 스텝 반복) ────────────────┐
 │                                                                 │
 │  1. Verification (이전 스텝의 draft 토큰 검증)                    │
 │     ├─ offloader->wait_and_load_layer(0)                        │
 │     ├─ llama_decode(ctx_tgt, batch_tgt)                         │
 │     │    └─ eval_callback이 매 layer마다:                        │
 │     │         ├─ 현재 layer SAVE (GPU→Host→SSD)                  │
 │     │         ├─ il+m번째 layer PREFETCH 큐잉 (SSD→Host)         │
 │     │         └─ il+1번째 layer LOAD (Host→GPU)                  │
 │     └─ backup_data = cb_data.data (hidden state 추출)            │
 │                                                                  │
 │  2. Accept/Reject 판정                                           │
 │     ├─ greedy: argmax 비교                                       │
 │     └─ stochastic: p_tgt/p_dft 비율 검증                         │
 │                                                                  │
 │  3. Recompute (draft 모델 KV cache 정리 + 재계산)                 │
 │     ├─ offloader->reset_layer_tracking()                         │
 │     ├─ offloader->prepare_target_pass()   ← 다음 verify용 prefetch│
 │     ├─ llama_memory_seq_rm/keep/cp (target & draft KV 정리)      │
 │     └─ llama_decode_eagle(ctx_dft, ...) (draft 모델 재계산)       │
 │                                                                  │
 │  4. Tree Drafting (draft 모델로 트리 구성)                        │
 │     ├─ depth 0..4까지 반복                                       │
 │     │    ├─ skip 판정 (어떤 시퀀스를 pruning 할지)                 │
 │     │    ├─ sampling + split (f_max에 따라 분기)                  │
 │     │    ├─ batch_tgt에 draft 토큰 추가 (target 검증용)           │
 │     │    ├─ batch_dft에 draft 토큰 추가 (다음 depth용)            │
 │     │    └─ llama_decode_eagle(ctx_dft, batch_dft, temp)          │
 │     └─ → 1번으로 돌아감                                          │
 │                                                                  │
 └──────────────────────────────────────────────────────────────────┘
```

---

## 3. Memory Pool 설계

### 3.1 pool_filter / pool_reuse 콜백

`llama_kv_cache`의 기존 layer 할당 메커니즘을 재활용하기 위해 두 가지 콜백을 생성자에 전달한다:

```cpp
// llama-kv-cache-offloaded.cpp:40-51

static llama_kv_cache::layer_filter_cb make_pool_filter(uint32_t n_pool_slots) {
    return [n_pool_slots](uint32_t il) -> bool {
        return il < n_pool_slots;   // layer 0..m-1만 KV 텐서 할당
    };
}

static llama_kv_cache::layer_reuse_cb make_pool_reuse(uint32_t n_pool_slots) {
    return [n_pool_slots](int32_t il) -> int32_t {
        if ((uint32_t)il < n_pool_slots) return -1;   // layer 0..m-1: 자체 텐서 소유
        return (int32_t)((uint32_t)il % n_pool_slots); // layer m+: slot = il % m (순환)
    };
}
```

**동작 원리:**
- `pool_filter`: layer index가 `m` 미만인 layer만 실제 GPU 텐서를 생성한다. 즉, GPU에는 물리적으로 **m개의 K,V 텐서 쌍**만 존재한다.
- `pool_reuse`: layer `il`이 어떤 물리적 slot의 텐서를 사용할지 결정한다.
  - layer 0..m-1: `return -1` → 자신만의 전용 텐서 (slot 0 = layer 0의 텐서, slot 1 = layer 1의 텐서, ...)
  - layer m+: `return il % m` → 순환 매핑 (예: m=4일 때 layer 4→slot 0, layer 5→slot 1, layer 28→slot 0)

### 3.2 layer_to_slot 매핑

```cpp
// llama-kv-cache-offloaded.h:161
uint32_t layer_to_slot(uint32_t model_layer) const {
    return model_layer % m_pool_slots;
}
```

이 매핑은 모든 layer에 적용된다. layer 0은 slot 0, layer 1은 slot 1, ..., layer m은 다시 slot 0. **하나의 slot을 여러 layer가 시분할로 공유**한다는 것이 핵심이다.

### 3.3 시간에 따른 slot 사용 패턴 (m=4, L=32 예시)

```
Graph 실행 순서:
  Layer  0 → Slot 0 (자체 소유)
  Layer  1 → Slot 1 (자체 소유)
  Layer  2 → Slot 2 (자체 소유)
  Layer  3 → Slot 3 (자체 소유)
  Layer  4 → Slot 0 (layer 0과 공유 — 이전에 layer 0 데이터 SAVE 필요!)
  Layer  5 → Slot 1 (layer 1과 공유)
  ...
  Layer 28 → Slot 0
  Layer 29 → Slot 1
  Layer 30 → Slot 2
  Layer 31 → Slot 3
```

따라서 layer 4가 slot 0을 사용하기 **전에**, layer 0의 업데이트된 KV 데이터를 SSD에 저장하고, layer 4의 기존 KV 데이터를 SSD에서 로딩해야 한다. 이것이 `eval_callback`이 하는 일이다.

---

## 4. 3-Tier 데이터 파이프라인

데이터는 3개의 저장소 계층을 거친다:

```
GPU Tensor (빠름, 용량 제한)
     ↕  ggml_backend_tensor_get/set  (동기, 메인 스레드)
Host Buffer (CPU RAM, 중간 속도)
     ↕  pread/pwrite                 (비동기, I/O 워커 스레드)
SSD File (느림, 무제한 용량)
```

### 왜 2단계로 나눌까?

**스레드 안전성** 때문이다:
- `ggml_backend_tensor_get/set`은 GPU 동기화가 필요하므로 **메인 스레드**에서만 호출 가능
- `pread/pwrite`는 별도 스레드에서 호출 가능

따라서:
1. **SAVE**: `tensor_to_host_delta()` (메인 스레드, 동기) → I/O 큐에 SAVE 태스크 push → `ssd_write_delta()` (워커 스레드, 비동기)
2. **LOAD**: `queue_prefetch()` → I/O 큐에 LOAD 태스크 push → `ssd_read_layer()` (워커 스레드, 비동기) → `host_to_tensor()` (메인 스레드, 동기)

---

## 5. 클래스 구조

### 5.1 llama_kv_cache_offloaded (llama-kv-cache-offloaded.h)

`llama_kv_cache`를 상속하여 SSD offloading 기능을 추가한다.

#### 멤버 변수 요약

| 멤버 | 타입 | 설명 |
|-------|------|------|
| `m_pool_slots` | `uint32_t` | GPU pool slot 수 (m) |
| `m_total_layers` | `uint32_t` | 모델의 총 layer 수 (L) |
| `m_cache_dir` | `string` | SSD 캐시 파일 디렉토리 경로 |
| `m_kv_size` | `uint32_t` | KV cache cell 수 (시퀀스 길이 예산) |
| `m_type_k`, `m_type_v` | `ggml_type` | K, V 캐시의 양자화 타입 |
| `m_v_trans` | `bool` | V 캐시가 전치(transposed) 상태인지. `!flash_attn`일 때 true |
| `m_ssd_layers` | `vector<ssd_layer>` | 각 layer의 SSD 파일 디스크립터 및 크기 정보 [L개] |
| `m_host_bufs` | `vector<host_buffer>` | 각 slot의 Host RAM 버퍼 [m개] |
| `m_slots` | `deque<slot_state>` | 각 slot의 상태 추적 (resident layer, prefetch 상태 등) [m개] |
| `m_io_thread` | `thread` | 비동기 I/O 워커 스레드 |
| `m_io_queue` | `queue<io_task>` | I/O 태스크 큐 (LOAD/SAVE) |
| `m_user_cb` | `callback` | 사용자 콜백 (cb_get_hidden 등) |
| `m_is_prefill` | `bool` | 현재 prefill 모드인지 (SSD에 이전 데이터 없음) |
| `m_delta_head` | `uint32_t` | 현재 ubatch의 delta 시작 cell index |
| `m_delta_count` | `uint32_t` | 현재 ubatch의 delta cell 수 |

#### 내부 구조체

```cpp
struct ssd_layer {
    int    fd_k, fd_v;           // K, V 파일의 POSIX file descriptor
    size_t k_total_bytes;        // K 파일의 전체 크기
    size_t v_total_bytes;        // V 파일의 전체 크기
};

struct host_buffer {
    void * k_data, * v_data;     // malloc으로 할당된 CPU RAM 버퍼
    size_t k_size, v_size;       // 각 버퍼의 크기 (바이트)
};

struct slot_state {
    int32_t           resident_layer;       // 현재 이 slot에 있는 layer (-1 = 비어있음)
    atomic<bool>      prefetch_pending;     // SSD→Host 작업이 큐에 있음
    atomic<bool>      prefetch_complete;    // SSD→Host 작업 완료
    int32_t           prefetch_target_layer;// prefetch 중인 target layer
    atomic<bool>      save_pending;         // Host→SSD 작업 진행 중 (host buf 잠금)
    mutex             mtx;
    condition_variable cv;
};
```

### 5.2 kv_offload_cb_data (콜백 데이터)

```cpp
struct kv_offload_cb_data {
    llama_kv_cache_offloaded * offloader;   // offloader 인스턴스 포인터
    void * user_data;                        // 사용자 콜백 데이터 (callback_data*)
};
```

`eval_callback`이 `void * user_data`로 전달받아 offloader와 사용자 콜백 데이터 모두에 접근한다.

---

## 6. API 상세 설명

### 6.1 생성 / 소멸

#### `llama_kv_cache_offloaded(..., n_pool_slots, cache_dir)`

**생성자.** 아래 순서로 초기화:

1. 부모 클래스 `llama_kv_cache` 생성 → `pool_filter`와 `pool_reuse` 콜백 전달 → GPU에 m개의 K,V 텐서만 할당
2. 각 layer의 SSD 파일 크기 계산 (`k_total_bytes`, `v_total_bytes`)
   - K: `n_embd_k_gqa(il) × kv_size × type_size / blck_size`
   - V (v_trans=false): `n_embd_v_gqa(il) × kv_size × type_size / blck_size`
   - V (v_trans=true): `n_embd_v_gqa_max() × kv_size × type_size / blck_size`
     - **주의**: v_trans=true일 때 `n_embd_v_gqa_max()`를 사용. 이는 `llama_kv_cache`가 전치 V 텐서 생성 시 max 값을 쓰기 때문
3. `m_slots` 초기화 (deque 사용 — mutex/atomic 등 non-movable 타입 포함)
4. Host buffer 할당 (`alloc_host_buffers`)
5. SSD 파일 열기 (`open_ssd_files`) — ftruncate로 사전 할당
6. I/O 워커 스레드 시작

#### `~llama_kv_cache_offloaded()`

1. `m_shutdown = true`로 설정, I/O 스레드 종료 대기
2. SSD 파일 닫기
3. Host buffer 해제

### 6.2 Host Buffer 관리

#### `alloc_host_buffers()`

m개의 host buffer 할당. 각 buffer는 **layer 0의 K,V 크기**를 기준으로 할당.
(모든 layer가 동일한 크기라고 가정 — LLaMA 계열 모델에서 성립)

```
host_buf[0] ← 크기: layer_0.k_total_bytes + layer_0.v_total_bytes
host_buf[1] ← 동일
...
host_buf[m-1] ← 동일
```

#### `free_host_buffers()`

모든 host buffer를 `free()`로 해제.

### 6.3 SSD 파일 관리

#### `open_ssd_files()`

각 layer별로 K, V 파일 2개씩 생성/열기:
```
{cache_dir}/layer_0_K.bin, layer_0_V.bin
{cache_dir}/layer_1_K.bin, layer_1_V.bin
...
{cache_dir}/layer_{L-1}_K.bin, layer_{L-1}_V.bin
```
총 **2×L개** 파일. `O_RDWR | O_CREAT`로 열고, `ftruncate`로 전체 크기 사전 할당.

#### `close_ssd_files()`

모든 파일 디스크립터 닫기.

### 6.4 Tensor ↔ Host Buffer (메인 스레드 전용)

이 함수들은 **GPU 동기화가 보장된 메인 스레드**에서만 호출해야 한다.

#### `tensor_to_host(model_layer, slot_id)` — 전체 복사

```
GPU tensor[slot_id].K  →  host_buf[slot_id].k_data   (ggml_backend_tensor_get, 전체)
GPU tensor[slot_id].V  →  host_buf[slot_id].v_data   (ggml_backend_tensor_get, 전체)
```

**Prefill 모드**에서 사용: layer가 처음 계산되므로 전체 KV 데이터를 저장해야 한다.

#### `tensor_to_host_delta(model_layer, slot_id, cell_start, cell_count)` — Delta 복사

**Decode 모드**에서 사용: 새로 추가된 cell만 복사하여 bandwidth 절약.

- **K (항상 row-major)**: `[cell_start × k_row .. (cell_start + cell_count) × k_row)` 영역만 복사
  ```
  k_row = n_embd_k_gqa × type_size / blck_size
  k_off = cell_start × k_row
  k_len = cell_count × k_row
  ggml_backend_tensor_get(k, buf.k_data + k_off, k_off, k_len)
  ```

- **V (v_trans=false)**: K와 동일한 row-major layout. delta 슬라이스만 복사.

- **V (v_trans=true)**: 전치 레이아웃 `[n_embd_v, kv_size]`. cell들이 embedding 차원마다 stride되어 분산.
  delta만 추출하려면 `n_embd_v`번의 작은 DMA 전송이 필요해서 비효율적.
  **대신 전체 V 텐서를 한 번에 복사**:
  ```
  ggml_backend_tensor_get(v, buf.v_data, 0, buf.v_size)
  ```
  이것은 정확성을 위한 trade-off — host buffer에는 전체 데이터가 있지만, SSD에는 delta만 기록한다.

#### `host_to_tensor(model_layer, slot_id)` — Host → GPU 전체 복사

```
host_buf[slot_id].k_data  →  GPU tensor[slot_id].K   (ggml_backend_tensor_set, 전체)
host_buf[slot_id].v_data  →  GPU tensor[slot_id].V   (ggml_backend_tensor_set, 전체)
```

### 6.5 SSD I/O (워커 스레드에서 실행)

#### `ssd_read_layer(model_layer, slot_id)` — SSD → Host 전체 읽기

```
pread(fd_k, host_buf[slot_id].k_data, k_total_bytes, offset=0)
pread(fd_v, host_buf[slot_id].v_data, v_total_bytes, offset=0)
```

항상 **전체 파일**을 읽는다. 파일 하나가 하나의 layer 전체 KV 데이터.

#### `ssd_write_delta(model_layer, slot_id, cell_start, cell_count)` — Host → SSD Delta 쓰기

- **K**: delta 영역만 `pwrite`
  ```
  k_offset = cell_start × k_row_bytes
  k_delta  = cell_count × k_row_bytes
  pwrite(fd_k, host_buf + k_offset, k_delta, k_offset)
  ```

- **V (v_trans=false)**: K와 동일하게 delta 영역만 기록

- **V (v_trans=true)**: 전치 레이아웃이므로 **embedding 차원별로** delta 기록:
  ```
  for j = 0..n_embd_v-1:
      offset = (cell_start + j × kv_size) × el_size
      delta  = cell_count × el_size
      pwrite(fd_v, host_buf + offset, delta, offset)
  ```
  이렇게 하면 host buffer에 전체 V 데이터가 있어도, SSD에는 **변경된 cell에 해당하는 영역만** 기록된다.

### 6.6 비동기 I/O 엔진

#### `io_worker_loop()` — 워커 스레드 메인 루프

```
while (!shutdown) {
    mutex lock → wait until queue not empty
    task = queue.front(); queue.pop()
    if task.op == LOAD: execute_load(task)
    else:               execute_save(task)
}
```

단일 워커 스레드가 큐의 태스크를 **순차적으로** 처리한다.

#### `execute_load(task)`

1. `ssd_read_layer(model_layer, slot_id)` — SSD → Host buffer
2. 해당 slot의 `prefetch_complete = true`, `prefetch_pending = false` 설정
3. `cv.notify_all()` — 메인 스레드가 `wait_and_load_layer()`에서 대기 중이면 깨움
4. `m_stats.loads++`

#### `execute_save(task)`

1. `ssd_write_delta(model_layer, slot_id, cell_start, cell_count)` — Host buffer → SSD
2. 해당 slot의 `save_pending = false` 설정
3. `cv.notify_all()` — 다음 save 대기 중인 eval_callback 깨움
4. `m_stats.saves++`

### 6.7 Prefetch / Save Orchestration

#### `queue_prefetch(model_layer)`

SSD → Host buffer 비동기 로딩을 큐에 추가.

1. 이미 해당 layer로 pending/complete이면 중복 방지를 위해 리턴
2. slot 상태 리셋: `prefetch_pending = true`, `prefetch_complete = false`
3. LOAD 태스크를 I/O 큐에 push

#### `is_prefetch_ready(model_layer)` — non-blocking 체크

해당 layer의 prefetch가 완료되었는지 확인. `prefetch_complete && target_layer == model_layer`.

#### `wait_and_load_layer(model_layer)` — blocking Host → GPU

1. **이미 GPU에 올바른 데이터가 있으면** (`resident_layer == model_layer` && not pending): 즉시 리턴 (HIT)
2. **Prefetch가 아직 완료되지 않았으면**: `cv.wait()`로 대기 (STALL)
3. Prefetch 완료 후: `host_to_tensor(model_layer, slot)` — Host buffer → GPU 텐서 복사
4. `resident_layer = model_layer` 업데이트

**중요**: 이 함수는 **메인 스레드에서만** 호출해야 한다 (`ggml_backend_tensor_set`이 GPU 동기화 필요).

#### `save_layer_delta(model_layer, cell_start, cell_count)`

1. `tensor_to_host_delta()` — GPU → Host (메인 스레드, 동기)
2. SAVE 태스크를 I/O 큐에 push (비동기)

#### `prepare_target_pass()`

Draft phase 시작 시 호출. 다음 target verification decode를 위해 **미리 prefetch 시작**.

1. `reset_layer_tracking()` — 이전 graph의 layer 추적 리셋
2. Prefill 모드가 아니면: layers 0..m-1에 대해 `queue_prefetch()` 호출

**왜 0..m-1만?** 이 layer들은 graph 실행 초반에 필요하다. layer m 이후는 eval_callback에서 실행 중 동적으로 prefetch가 큐잉된다.

#### `on_verify_complete(n_accepted)`

Target verification 완료 후 호출. `m_is_prefill = false` 설정. (첫 verify 이후부터 prefill 모드 해제)

### 6.8 콜백 관련

#### `set_user_callback(cb, data)`

사용자 콜백(예: `cb_get_hidden`)을 설정. eval_callback이 이 콜백을 **합성(compose)**하여 호출한다.

#### `set_prefill(bool)`

Prefill 모드 플래그. true일 때는 eval_callback이 전체 tensor를 저장하고, next layer load를 건너뛴다.

#### `reset_layer_tracking()`

`m_last_completed_layer = -1`. 새 graph 실행 시작 시 리셋.

#### `set_delta_info(head, count)`

현재 ubatch의 KV cache 변경 범위를 설정:
- `head`: 새 토큰이 기록되기 시작하는 cell index
- `count`: 새로 기록되는 토큰 수

**호출 위치**: `llama_kv_cache.cpp`의 `apply()` 함수 내부에서 자동 호출:

```cpp
// src/llama-kv-cache.cpp (수정됨)
auto * offloaded = dynamic_cast<llama_kv_cache_offloaded *>(kv);
if (offloaded) {
    offloaded->set_delta_info(sinfos[i_cur].head(), (uint32_t)sinfos[i_cur].size());
}
```

이 정보는 eval_callback이 `tensor_to_host_delta()`에 전달하여 **필요한 cell만** GPU→Host 복사한다.

---

## 7. eval_callback 동작 흐름

### 7.1 콜백 등록

```cpp
// eagle-KV-offloading.cpp:209
ctx_tgt->set_eval_callback(llama_kv_cache_offloaded::eval_callback, &offload_cb_data);
```

GGML graph 스케줄러가 각 텐서 연산 전후에 이 콜백을 호출한다.

### 7.2 ask == true (실행 전 질의)

```
텐서 이름 확인:
├─ "kqv_out-{il}": return true   → 이 텐서 인터셉트 (실행 후 콜백 재호출)
├─ "result_norm":  cb_get_hidden에 위임 → return true
└─ 그 외:          cb_get_hidden에 위임 → return false (인터셉트 안 함)
```

### 7.3 ask == false (실행 완료)

```
텐서 이름 확인:
├─ "result_norm":
│    └─ cb_get_hidden(tensor, false, user_data) 호출 → hidden state 추출
│       cb_data.data에 result_norm 텐서 데이터 복사
│       → return true
│
├─ "kqv_out-{il}" (layer il 완료):
│    │
│    ├─ [Decode 모드] (m_is_prefill == false):
│    │    ① save_pending 대기 (이전 SAVE가 host_buf 사용 중이면 기다림)
│    │    ② save_pending = true (host_buf 잠금)
│    │    ③ tensor_to_host_delta(il, slot, delta_head, delta_count)
│    │       → GPU → Host buffer (delta만)
│    │    ④ SAVE 태스크 큐잉 (Host → SSD, 비동기)
│    │    ⑤ queue_prefetch(il + m)
│    │       → il+m번째 layer (같은 slot을 다음에 사용할 layer) SSD→Host 예약
│    │
│    ├─ [Prefill 모드] (m_is_prefill == true):
│    │    ① save_pending 대기
│    │    ② save_pending = true
│    │    ③ tensor_to_host(il, slot)
│    │       → GPU → Host buffer (전체)
│    │    ④ SAVE 태스크 큐잉 (cell_start=0, cell_count=kv_size)
│    │    ⑤ (prefetch 없음 — 다음 layer는 처음 계산되므로 load 불필요)
│    │
│    └─ [공통] 다음 layer 로딩:
│         if il+1 < L && !prefill:
│             wait_and_load_layer(il + 1)
│             → Host → GPU (il+1번째 layer 데이터를 GPU 텐서에 적재)
│
└─ 그 외: return true (무시)
```

### 7.4 콜백 타이밍 시퀀스 (m=4, L=32, Decode 모드)

```
prepare_target_pass() → queue LOAD(0), LOAD(1), LOAD(2), LOAD(3)
wait_and_load_layer(0) → Host→GPU slot 0 (layer 0)

Graph 실행 시작:
  Layer 0 attention → uses slot 0 ✓
  kqv_out-0 callback:
    SAVE layer 0 delta (GPU→Host→SSD 비동기)
    queue LOAD(4)
    wait_and_load_layer(1) → Host→GPU slot 1 (layer 1)
  
  Layer 1 attention → uses slot 1 ✓
  kqv_out-1 callback:
    SAVE layer 1 delta
    queue LOAD(5)
    wait_and_load_layer(2) → Host→GPU slot 2 (layer 2)
  
  ...계속...
  
  Layer 31 attention → uses slot 3 ✓
  kqv_out-31 callback:
    SAVE layer 31 delta
    (layer 32 없으므로 prefetch 없음)
    (layer 32 없으므로 load 없음)
  
  result_norm callback:
    cb_get_hidden → hidden state 추출 → cb_data.data에 저장
```

---

## 8. 애플리케이션 흐름 — eagle-KV-offloading.cpp

### 8.1 커맨드라인 인수

```
--kv-offload-slots <m>     GPU pool slot 수 (기본값: 16)
--kv-offload-dir <dir>     SSD 캐시 디렉토리 (기본값: /tmp/kv_offload)
--tree-budget <7|25>       트리 구조 선택 (기본값: 7)
```

이 인수들은 `common_params_parse()` 전에 미리 파싱 후 제거된다 (llama.cpp 기본 파서가 모르는 인수).

### 8.2 초기화 순서

```
1. Target 모델 로드 (common_init_from_params)
   - params.cb_eval = cb_get_hidden (hidden state 추출용)
   - GPU에 전체 L layer KV cache 할당됨

2. KV Offloader 설치:
   a. ctx_tgt->set_memory(nullptr)
      → 기존 전체 KV cache 해제 (GPU 메모리 회수)
   b. llama_kv_cache_offloaded 생성 (m slot만 GPU에 할당)
   c. offloader->set_user_callback(cb_get_hidden, &cb_data)
      → eval_callback이 result_norm을 cb_get_hidden에 전달하도록 합성
   d. ctx_tgt->set_memory(offloader_ptr)
      → 컨텍스트의 메모리 모듈 교체
   e. ctx_tgt->set_eval_callback(eval_callback, &offload_cb_data)
      → GGML 스케줄러에 합성 콜백 등록

3. Draft 모델 로드
   - params.cb_eval = cb_get_hidden (offloading 콜백 아님)
   - Draft 모델은 일반 KV cache 사용 (offloading 없음)

4. LM Head Sharing
   - draft_model->output = target_model->output
   - Draft 모델의 output 텐서를 target의 것으로 교체
   - ⚠ 참고: output_norm은 현재 공유하지 않음
```

### 8.3 Prefill

```cpp
offloader->set_prefill(true);                               // (1)

// 프롬프트의 처음 ~ 마지막-1번째 토큰을 한 번에 decode
llama_decode(ctx_tgt, temp_batch_tgt);                      // (2)
// → eval_callback이 각 layer의 전체 KV를 SSD에 저장
// → result_norm 콜백으로 hidden state 추출 → sliced_data

offloader->set_prefill(false);                              // (3)
offloader->prepare_target_pass();                           // (4)
offloader->wait_and_load_layer(0);                          // (5)

// 마지막 토큰 decode (1개 토큰)
llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1)); // (6)
// → eval_callback이 decode 모드로 동작:
//   각 layer: delta save → prefetch next → load next
// → backup_data에 hidden state 저장

// Draft 모델 prefill
llama_decode_eagle(ctx_dft, ..., sliced_data.data());       // (7)
```

**단계별 설명:**

1. Prefill 모드 켜기: eval_callback이 전체 tensor를 저장하고, next layer load를 건너뜀 (처음이라 모든 layer가 fresh compute)
2. Target 모델 prefill decode. 각 layer 실행 후 eval_callback이 KV 데이터를 SSD에 저장.
   - Prefill 완료 후, GPU slot에는 **마지막으로 각 slot을 사용한 layer의 데이터**가 남아있음 (예: slot 0 = layer 28의 데이터)
3. Decode 모드 전환. 이후부터 eval_callback은 delta save + next layer load 수행.
4. `prepare_target_pass()`: layers 0..m-1의 SSD→Host prefetch를 비동기로 시작
5. `wait_and_load_layer(0)`: layer 0의 prefetch 완료를 기다리고 Host→GPU 복사. 이제 slot 0에는 layer 0의 올바른 데이터가 있음.
6. 마지막 토큰 decode. eval_callback이 layer별 save/load를 수행하여 모든 layer를 올바르게 처리.
7. Draft 모델 prefill (EAGLE의 hidden state 전달)

### 8.4 Verification Loop

```cpp
// target verification decode 전:
offloader->wait_and_load_layer(0);         // layer 0 GPU에 로딩
llama_decode(ctx_tgt, batch_tgt);          // target 모델 decode
// → eval_callback이 layer별 save/load 처리
// → result_norm → cb_get_hidden → cb_data.data
backup_data = cb_data.data;                // hidden state 백업
```

### 8.5 Recompute (Verification 후 Draft 재계산)

```cpp
offloader->reset_layer_tracking();
offloader->prepare_target_pass();          // 다음 verification용 prefetch 시작

// Draft KV cache 정리
llama_memory_seq_keep(mem_dft, s_keep);    // accepted 시퀀스만 보존
llama_memory_seq_rm(mem_dft, 0, recompute_point, -1);

// Target KV cache 정리  
llama_memory_seq_rm(mem_tgt, s_keep, n_past_tgt, -1);
llama_memory_seq_keep(mem_tgt, s_keep);
llama_memory_seq_cp(mem_tgt, s_keep, 0, -1, -1);
llama_memory_seq_keep(mem_tgt, 0);

// Draft 모델 재계산 (accepted 토큰들)
if (i_dft > 0) {
    llama_decode_eagle(ctx_dft, batch_dft, temp4.data());
}
llama_decode_eagle(ctx_dft, batch_dft, temp3.data());
```

**`prepare_target_pass()`가 recompute 앞에 호출되는 이유**: Draft 모델 recompute에는 시간이 걸린다. 이 시간 동안 I/O 워커가 layers 0..m-1을 SSD에서 Host로 비동기 로딩할 수 있다. Draft phase(tree decoding)가 끝나고 target verification이 시작될 때 이미 prefetch가 완료되어 있을 가능성이 높다.

---

## 9. Tree Decoding 로직 (Budget-7 / Budget-25)

### 9.1 트리 파라미터

```cpp
#define n_depth 5       // 트리 최대 깊이
#define expand_k 2      // (정의만 존재, 실제 사용 안 됨)
#define rerank_k 10     // (정의만 존재, 실제 사용 안 됨)

int third_depth[4] = { 0, 1, 0, 0 };   // budget-7
// budget-25: third_depth = { 0, 1, 4, 5 }
```

`third_depth`는 특정 depth에서 **어떤 시퀀스 ID를 계속 draft할지** 결정하는 배열.

### 9.2 Budget-7 트리 구조 (n_seq_dft=2 기준)

```
Depth 0:  [seq 0] ──┬── top-1 → seq 0 계속
                     └── top-2 → seq 1 생성 (split)
Depth 1:  [seq 0] ── top-1 (f_max=1, 확장 없음)
          [seq 1] ── (f_max=0, skip)
Depth 2:  [seq 0] ── top-1
Depth 3:  [seq 0] ── top-1
Depth 4:  [seq 0] ── top-1

총 draft 토큰: 7개 (batch_tgt에 추가됨)
```

| depth | skip 조건 | f_max | batch_dft 추가 조건 |
|-------|----------|-------|-------------------|
| 0 | 없음 | 2 | 모든 seq |
| 1 | 없음 | s==0: 1, else: 0 | s ∈ third_depth[0..1] |
| 2 | s ∉ third_depth → skip | s==0: 1, else: 0 | s ∈ third_depth[0..1] |
| 3 | s ∉ third_depth → skip | s==0: 1, else: 0 | s==0만, else drafting=false |
| 4 | s ∉ third_depth → skip | 1 | s==0만, else drafting=false |

### 9.3 Budget-25 트리 구조

```
Depth 0:  [seq 0] → 4개 분기 (f_max=4) → seq 0,1,2,3
Depth 1:  [seq 0] → 3개 (f_max=3), [seq 1] → 2개, [seq 2] → 2개, [seq 3] → 1개
Depth 2:  [seq 0] → 3개, [seq 1] → 1개, [seq 4] → 2개, [seq 5] → 2개
Depth 3:  [seq 0] → 3개
Depth 4:  f_max=2

총 draft 토큰: ~25개
```

### 9.4 핵심 로직 흐름 (한 depth 반복)

```
for each depth i = 0..n_draft-1:
    1. skip 판정: 어떤 시퀀스를 이번 depth에서 제외할지
    2. for each active, non-skipped seq s:
        a. common_sampler_sample() → top-k 토큰 추출
        b. hidden state 수집 (cb_data → temp)
        c. f_max 결정 → split (새 시퀀스 생성)
        d. 각 (원본 + split) 시퀀스에 대해:
           - batch_tgt에 토큰 추가 (target verification용)
           - 조건부로 batch_dft에 토큰 추가 (다음 depth용)
    3. llama_decode_eagle(ctx_dft, batch_dft, temp) → draft 모델 decode
    4. cur_depth++
```

---

## 10. Delta Save 최적화

### 10.1 왜 필요한가

매 target decode에서 KV cache에 새로 기록되는 cell은 보통 **1~수개** (batch_tgt의 토큰 수). 전체 KV cache를 매번 저장하면 bandwidth 낭비.

### 10.2 Delta 정보 전파 경로

```
llama_kv_cache_context::apply()
  └─ sinfos[i_cur].head()  → delta 시작 위치
     sinfos[i_cur].size()  → delta cell 수
  └─ offloaded->set_delta_info(head, size)
       → m_delta_head, m_delta_count 설정

eval_callback (kqv_out-{il}):
  └─ tensor_to_host_delta(il, slot, m_delta_head, m_delta_count)
       → GPU의 delta 영역만 Host buffer에 복사
  └─ ssd_write_delta(il, slot, m_delta_head, m_delta_count)
       → Host buffer의 delta 영역만 SSD에 기록
```

### 10.3 V 전치 (v_trans=true) 시 특수 처리

V가 전치되면 레이아웃이 `[n_embd_v, kv_size]`이다. cell `c`의 데이터가 embedding 차원마다 `kv_size`간격으로 분산되어 있어, delta 슬라이스 추출이 복잡하다.

- **GPU → Host** (`tensor_to_host_delta`): V 전체 복사 (한 번의 큰 DMA가 n_embd_v번의 작은 DMA보다 효율적)
- **Host → SSD** (`ssd_write_delta`): embedding 차원별로 delta cell 영역만 기록 (총 n_embd_v번의 pwrite)
- **SSD → Host** (`ssd_read_layer`): 항상 전체 읽기

---

## 11. 동기화 메커니즘

### 11.1 save_pending 플래그

```
eval_callback(kqv_out-{il}):
  1. WAIT: cv.wait(lock, [&] { return !save_pending; })
     → 이전 SAVE가 host_buf를 사용 중이면 대기
  2. save_pending = true
     → host_buf 잠금 (이제 tensor_to_host_delta가 host_buf에 쓸 수 있음)
  3. tensor_to_host_delta() → host_buf에 GPU 데이터 복사
  4. SAVE 태스크 큐잉

execute_save() (워커 스레드):
  1. ssd_write_delta() → host_buf 읽기 → SSD 기록
  2. save_pending = false → host_buf 잠금 해제
  3. cv.notify_all()
```

**왜 필요한가**: 같은 slot을 두 layer가 연속으로 사용할 때 (예: layer 0 → layer 4), layer 0의 SAVE가 host_buf에서 SSD로 기록하는 동안 layer 4의 eval_callback이 host_buf를 덮어쓰면 데이터 손상.

### 11.2 prefetch_pending / prefetch_complete 플래그

```
queue_prefetch(layer):
  prefetch_pending = true, prefetch_complete = false
  LOAD 태스크 큐잉

execute_load() (워커 스레드):
  ssd_read_layer() → host_buf에 SSD 데이터 복사
  prefetch_complete = true, prefetch_pending = false
  cv.notify_all()

wait_and_load_layer(layer):
  if resident_layer == layer: return (HIT)
  cv.wait(lock, [&] { return prefetch_complete; })  (STALL)
  host_to_tensor() → host_buf → GPU 텐서
  resident_layer = layer
```

### 11.3 I/O 큐 동기화

```
m_io_mutex + m_io_cv:
  - 큐잉: lock(m_io_mutex) → push → notify_one
  - 워커: wait(m_io_cv, [&] { !queue.empty() || shutdown }) → pop → 처리
```

---

## 12. 알려진 이슈 및 주의사항

### 12.1 Prefill 후 Stale Data

Prefill 완료 후 각 GPU slot에는 **마지막으로 해당 slot을 사용한 layer의 데이터**가 남아있다 (예: m=4이면 slot 0에 layer 28 데이터).

**해결**: `set_prefill(false)` → `prepare_target_pass()` → `wait_and_load_layer(0)` 순서로 호출하여 slot 0에 layer 0의 올바른 데이터를 로딩.

### 12.2 v_trans=true에서의 SSD Write Overhead

V 전치 상태에서 `ssd_write_delta()`가 embedding 차원마다 별도 `pwrite()`를 호출 (총 n_embd_v회). n_embd_v가 크면(예: 4096) 시스템 콜 overhead 발생.

**잠재적 개선**: host buffer에서 delta 영역을 연속 버퍼로 재배치한 뒤 한 번에 pwrite, 또는 vectored I/O (`writev`) 사용.

### 12.3 output_norm 공유 미구현

`speculative-eagle.cpp`의 활성 코드(budget-25)에서는:
```cpp
if (llama_get_model(ctx_tgt)->output_norm && !llama_get_model(ctx_dft)->output_norm) {
    const_cast<llama_model *>(llama_get_model(ctx_dft))->output_norm =
        llama_get_model(ctx_tgt)->output_norm;
}
```

`eagle-KV-offloading.cpp`에서는 이 부분이 **누락**되어 있음. EAGLE draft 모델에 자체 `output_norm`이 없는 경우 draft logit이 달라질 수 있다.

### 12.4 Host Buffer 크기 가정

`alloc_host_buffers()`가 **layer 0의 크기**를 모든 slot에 동일하게 사용:
```cpp
size_t k_sz = m_ssd_layers[0].k_total_bytes;
size_t v_sz = m_ssd_layers[0].v_total_bytes;
```

**가정**: 모든 layer의 K, V 크기가 동일 (LLaMA 계열에서 성립). 다른 아키텍처에서는 layer별 크기가 다를 수 있으므로 max를 사용해야 할 수 있음.

### 12.5 단일 I/O 워커 스레드

현재 I/O 워커가 1개이므로 SAVE와 LOAD가 직렬 처리. SAVE가 많으면 후속 LOAD가 지연될 수 있다.

**잠재적 개선**: SAVE/LOAD 별도 큐 + 별도 워커, 또는 우선순위 큐 (LOAD 우선).

### 12.6 SSD 파일 재사용

SSD 파일은 `O_CREAT`으로 열지만 이전 실행의 데이터가 남아있을 수 있다. Prefill에서 전체 덮어쓰므로 문제없지만, prefill 없이 시작하면 stale 데이터 위험.

### 12.7 accept length 차이

`llama-eagle-kv-offloading --tree-budget 7`과 `llama-speculative-eagle`(budget-7 활성화)의 accept length가 다를 수 있는 원인:

1. **12.3의 output_norm 공유 차이** (budget-25 활성 코드 기반으로 수정한 경우)
2. **KV offloading으로 인한 GPU 동기화 패턴 변화** → 부동소수점 비결정성 가능성
3. **tree 로직 자체는 `-np 2`에서 동일함** (확인 완료)

---

## 부록: 실행 예시

```bash
./llama-eagle-kv-offloading \
    -m /path/to/target-model.gguf \
    -md /path/to/eagle-draft-model.gguf \
    -f prompt.txt \
    -ngl 40 -ngld 40 \
    -c 0 --color \
    --top-k 2 --top-p 1.0 --min-p 0.0 --temp 0.0 \
    --draft-max 7 --draft-min 1 \
    --n-predict 100 \
    -np 2 -s 1234 \
    -kvu --no-mmap \
    --kv-offload-slots 4 \
    --kv-offload-dir ./kv_dir \
    --tree-budget 7
```

### 출력 예시

```
KV offloading: 4 pool slots, dir: ./kv_dir, tree-budget: 7
[KV-Offload] Initializing: 4 pool slots, 32 total layers, kv_size=8192
[KV-Offload] Allocated 4 host buffers (64.00 MB each)
[KV-Offload] Ready. Per-layer K size: 32.00 MB, V size: 32.00 MB
KV offloader installed: 4 pool slots / 32 model layers
LM head sharing: OK

... (생성된 텍스트) ...

KV offload stats:
  prefetch hits  : 1234
  prefetch stalls: 56
  saves          : 3200
  loads          : 3200

============================================================
       EAGLE + KV SSD Offloading  Performance Summary
============================================================
  Pool slots     : 4 / 32 model layers
  Prefill        :   128 tokens |    234.56 ms |   545.89 t/s
  Decode         :   100 tokens |   5678.90 ms |    17.61 t/s
  Decode latency :              |     56.79 ms/tok
------------------------------------------------------------
  Draft length          : 5.000
  Avg accept length     : 2.345
  Accept ratio          : 46.900%
------------------------------------------------------------
  Avg draft phase       :    12.345 ms
  Avg verification      :    43.210 ms
  Avg T_d (1-tok dft)   :     3.456 ms
============================================================
```
