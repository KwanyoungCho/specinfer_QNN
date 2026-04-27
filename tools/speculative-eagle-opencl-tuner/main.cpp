#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef SPECINFER_QNN_OPENCL_KERNEL_DIR
#define SPECINFER_QNN_OPENCL_KERNEL_DIR "."
#endif

namespace {

struct Options {
    int platform_index = 0;
    int device_index = 0;
    int n_scores = 128256;
    int src_rows = -1;
    int top_k = 64;
    int hidden_dim = 4096;
    int lmhead_batch = 1;
    int gather_rows = -1;
    int min_gather_rows = 512;
    int warmup = 1;
    int iters = 5;
    int seed = 42;
    std::string kernel_dir;
    std::string ids_file;
    bool tune_topk = true;
    bool tune_bucket_topk = true;
    bool tune_id_sort = true;
    bool tune_gather = true;
    bool tune_indexed = true;
    bool allow_non_power_of_two_local = false;
    bool verbose = false;
    bool show_help = false;
};

struct OneDimResult {
    size_t lws = 0;
    double avg_ms = std::numeric_limits<double>::infinity();
    double min_ms = std::numeric_limits<double>::infinity();
};

struct TwoDimResult {
    size_t lx = 0;
    size_t ly = 0;
    double avg_ms = std::numeric_limits<double>::infinity();
    double min_ms = std::numeric_limits<double>::infinity();
};

struct GatherTuningConfig {
    size_t lx = 0;
    size_t ly = 0;
    int rows_per_thread = 1;
    int k4_per_thread = 1;
    bool use_local_ids = false;
    bool legacy_kernel = false;
};

struct GatherResult {
    GatherTuningConfig config;
    double avg_ms = std::numeric_limits<double>::infinity();
    double min_ms = std::numeric_limits<double>::infinity();
};

struct IndexedMatvecTuningConfig {
    size_t local_size = 64;
    int rows_per_subgroup = 8;
    bool abi_tile = false;
    size_t wi_m = 0;
    size_t wi_k = 0;
    int abi_n_tile = 8;
    bool abi_no_split = false;
    int abi_m_tile = 4;
    bool abi_local_b = false;
    bool abi_prefetch = false;
};

struct MatvecResult {
    std::string label;
    IndexedMatvecTuningConfig indexed_config;
    double avg_ms = std::numeric_limits<double>::infinity();
    double min_ms = std::numeric_limits<double>::infinity();
};

static constexpr uint32_t kPatternSeed0Q = 1103515245u;
static constexpr uint32_t kPatternSeed1Q = 12345u;
static constexpr uint16_t kHalfOneBits = 0x3c00u;

static constexpr int32_t kBucketBits   = 11;
static constexpr int32_t kBucketCount  = 1 << kBucketBits;
static constexpr int32_t kBucketShift  = 32 - kBucketBits;

static const char * kPatternInitKernels = R"CLC(
kernel void kernel_init_pattern_u16(
        global ushort * dst,
        const uint      major_dim,
        const uint      minor_dim,
        const uint      seed0,
        const uint      seed1) {
    const ulong gid = get_global_id(0);
    const ulong total = (ulong) major_dim * (ulong) minor_dim;
    if (gid >= total) {
        return;
    }

    const uint major = (uint) (gid / (ulong) minor_dim);
    const uint minor = (uint) (gid - (ulong) major * (ulong) minor_dim);

    uint value = major * seed0;
    value ^= minor * seed1;
    value ^= (major + 17u) * (minor + 31u);

    dst[gid] = (ushort) (value & 0xffffu);
}
)CLC";

static const char * kMinimalArgsortKernels = R"CLC(
kernel void kernel_init_i32_range(
    global int * dst,
    const int    n_valid,
    const int    n_padded,
    const int    pad_value
) {
    const int gid = get_global_id(0);
    if (gid >= n_padded) {
        return;
    }

    dst[gid] = gid < n_valid ? gid : pad_value;
}

kernel void kernel_fill_i32(
    global int * dst,
    const int    n,
    const int    value
) {
    const int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    dst[gid] = value;
}

inline int should_swap_desc_f32_i32(const float lhs_value, const int lhs_index, const float rhs_value, const int rhs_index) {
    return lhs_value < rhs_value || (lhs_value == rhs_value && lhs_index > rhs_index);
}

inline int should_swap_asc_f32_i32(const float lhs_value, const int lhs_index, const float rhs_value, const int rhs_index) {
    return lhs_value > rhs_value || (lhs_value == rhs_value && lhs_index > rhs_index);
}

kernel void kernel_bitonic_sort_step_f32_i32(
    global float * values,
    global int   * indices,
    const int      n_padded,
    const int      stage_j,
    const int      stage_k
) {
    const uint i = get_global_id(0);
    if (i >= (uint) n_padded) {
        return;
    }

    const uint ixj = i ^ (uint) stage_j;
    if (ixj <= i || ixj >= (uint) n_padded) {
        return;
    }

    const float lhs_value = values[i];
    const float rhs_value = values[ixj];
    const int lhs_index = indices[i];
    const int rhs_index = indices[ixj];
    const int descending = ((i & (uint) stage_k) == 0);

    const int should_swap = descending
            ? should_swap_desc_f32_i32(lhs_value, lhs_index, rhs_value, rhs_index)
            : should_swap_asc_f32_i32(lhs_value, lhs_index, rhs_value, rhs_index);

    if (should_swap) {
        values[i] = rhs_value;
        values[ixj] = lhs_value;
        indices[i] = rhs_index;
        indices[ixj] = lhs_index;
    }
}

inline int should_swap_asc_i32(const int lhs_value, const int rhs_value) {
    return lhs_value > rhs_value;
}

inline int should_swap_desc_i32(const int lhs_value, const int rhs_value) {
    return lhs_value < rhs_value;
}

inline uint topk_radix_key_f32(float f) {
    uint u = as_uint(f);
    uint mask = (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return u ^ mask;
}

kernel void kernel_topk_hist_f32(
    global const float * scores,
    const int            n_scores,
    global uint *        hist,
    const int            n_buckets,
    const int            bucket_shift
) {
    const int gid = get_global_id(0);
    if (gid >= n_scores) {
        return;
    }
    const float s = scores[gid];
    uint key = topk_radix_key_f32(s);
    uint b = key >> bucket_shift;
    if (b >= (uint) n_buckets) {
        b = (uint) n_buckets - 1u;
    }
    atomic_inc(&hist[b]);
}

kernel void kernel_topk_compact_f32(
    global const float * scores,
    const int            n_scores,
    const int            bucket_shift,
    const uint           threshold_bucket,
    const uint           taken_above,
    const uint           quota_at,
    global uint *        counters,
    global int *         out_indices
) {
    const int gid = get_global_id(0);
    if (gid >= n_scores) {
        return;
    }
    const float s = scores[gid];
    uint key = topk_radix_key_f32(s);
    uint b = key >> bucket_shift;

    if (b > threshold_bucket) {
        uint slot = atomic_inc(&counters[0]);
        if (slot < taken_above) {
            out_indices[slot] = gid;
        }
    } else if (b == threshold_bucket) {
        uint slot = atomic_inc(&counters[1]);
        if (slot < quota_at) {
            out_indices[taken_above + slot] = gid;
        }
    }
}

kernel void kernel_bitonic_sort_step_i32(
    global int * values,
    const int   n_padded,
    const int   stage_j,
    const int   stage_k
) {
    const uint i = get_global_id(0);
    if (i >= (uint) n_padded) {
        return;
    }

    const uint ixj = i ^ (uint) stage_j;
    if (ixj <= i || ixj >= (uint) n_padded) {
        return;
    }

    const int lhs_value = values[i];
    const int rhs_value = values[ixj];
    const int ascending = ((i & (uint) stage_k) == 0);

    const int should_swap = ascending
            ? should_swap_asc_i32(lhs_value, rhs_value)
            : should_swap_desc_i32(lhs_value, rhs_value);

    if (should_swap) {
        values[i] = rhs_value;
        values[ixj] = lhs_value;
    }
}

)CLC";

static const char * kTunableGatherKernels = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#ifndef GATHER_ROWS_PER_THREAD
#define GATHER_ROWS_PER_THREAD 1
#endif

#ifndef GATHER_K4_PER_THREAD
#define GATHER_K4_PER_THREAD 1
#endif

#ifndef GATHER_USE_LOCAL_IDS
#define GATHER_USE_LOCAL_IDS 0
#endif

kernel void kernel_gather_rows_q4_0_tunable_i32(
        global const ushort * src_q,
        global const half   * src_d,
        global const int    * src_ids,
        global ushort       * dst_q,
        global half         * dst_d,
        int                   src_rows,
        int                   dst_rows,
        int                   k4_count,
        local int           * local_ids
) {
    const int lx = get_local_size(0);
    const int ly = get_local_size(1);
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);

    const int row_tile = lx * GATHER_ROWS_PER_THREAD;
    const int k4_tile = ly * GATHER_K4_PER_THREAD;

    const int row_base = (int) get_group_id(0) * row_tile;
    const int k4_base = (int) get_group_id(1) * k4_tile;

#if GATHER_USE_LOCAL_IDS
    const int linear_lid = lid_y * lx + lid_x;
    const int linear_size = lx * ly;
    for (int row_offset = linear_lid; row_offset < row_tile; row_offset += linear_size) {
        const int dst_row = row_base + row_offset;
        local_ids[row_offset] = dst_row < dst_rows ? src_ids[dst_row] : -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (int row_iter = 0; row_iter < GATHER_ROWS_PER_THREAD; ++row_iter) {
        const int dst_row = row_base + lid_x + row_iter * lx;
        if (dst_row >= dst_rows) {
            continue;
        }

#if GATHER_USE_LOCAL_IDS
        const int src_row = local_ids[lid_x + row_iter * lx];
#else
        const int src_row = src_ids[dst_row];
#endif

        for (int k4_iter = 0; k4_iter < GATHER_K4_PER_THREAD; ++k4_iter) {
            const int k4_idx = k4_base + lid_y + k4_iter * ly;
            if (k4_idx >= k4_count) {
                continue;
            }

            ushort q_value = (ushort) 0;
            half   d_value = (half) 0;

            if (src_row >= 0 && src_row < src_rows) {
                q_value = src_q[(ulong) k4_idx * (ulong) src_rows + (ulong) src_row];
                if ((k4_idx & 7) == 0) {
                    const int kb = k4_idx >> 3;
                    d_value = src_d[(ulong) kb * (ulong) src_rows + (ulong) src_row];
                }
            }

            dst_q[(ulong) k4_idx * (ulong) dst_rows + (ulong) dst_row] = q_value;
            if ((k4_idx & 7) == 0) {
                const int kb = k4_idx >> 3;
                dst_d[(ulong) kb * (ulong) dst_rows + (ulong) dst_row] = d_value;
            }
        }
    }
}
)CLC";

static const char * kQ4MatvecKernels = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK4_0 32
#define Q4_N_DST 8
#ifndef INDEXED_ROWS_PER_SG
#define INDEXED_ROWS_PER_SG 8
#endif
#ifdef cl_intel_required_subgroup_size
#define Q4_SIMD_HALF_WIDTH 8
#else
#define Q4_SIMD_HALF_WIDTH 32
#endif

inline float block_q4_0_dot_transposed(
        global const ushort * q,
        global const half   * d,
        int                   rows,
        int                   row,
        int                   ib,
        float                 sumy,
        float16               yl,
        int                   il) {
    const int k4 = ib*8 + il/2;
    const float ds = d[(ulong) ib * (ulong) rows + (ulong) row];
    const ushort q0 = q[(ulong) (k4 + 0) * (ulong) rows + (ulong) row];
    const ushort q1 = q[(ulong) (k4 + 1) * (ulong) rows + (ulong) row];
    const ushort q2 = q[(ulong) (k4 + 2) * (ulong) rows + (ulong) row];
    const ushort q3 = q[(ulong) (k4 + 3) * (ulong) rows + (ulong) row];

    float acc = 0.f;

    acc += yl.s0 * (q0 & 0x000F);
    acc += yl.s1 * (q0 & 0x0F00);
    acc += yl.s8 * (q0 & 0x00F0);
    acc += yl.s9 * (q0 & 0xF000);

    acc += yl.s2 * (q1 & 0x000F);
    acc += yl.s3 * (q1 & 0x0F00);
    acc += yl.sa * (q1 & 0x00F0);
    acc += yl.sb * (q1 & 0xF000);

    acc += yl.s4 * (q2 & 0x000F);
    acc += yl.s5 * (q2 & 0x0F00);
    acc += yl.sc * (q2 & 0x00F0);
    acc += yl.sd * (q2 & 0xF000);

    acc += yl.s6 * (q3 & 0x000F);
    acc += yl.s7 * (q3 & 0x0F00);
    acc += yl.se * (q3 & 0x00F0);
    acc += yl.sf * (q3 & 0xF000);

    return ds * (sumy * -8.f + acc);
}

inline void load_hidden_tile(
        global const float * yb,
        float16            * yl,
        float              * sumy) {
    *sumy = 0.f;
    *sumy += yb[0];
    *sumy += yb[1];
    *sumy += yb[2];
    *sumy += yb[3];
    *sumy += yb[4];
    *sumy += yb[5];
    *sumy += yb[6];
    *sumy += yb[7];

    *sumy += yb[16];
    *sumy += yb[17];
    *sumy += yb[18];
    *sumy += yb[19];
    *sumy += yb[20];
    *sumy += yb[21];
    *sumy += yb[22];
    *sumy += yb[23];

    yl->s0 = yb[0];
    yl->s1 = yb[1]/256.f;

    yl->s2 = yb[2];
    yl->s3 = yb[3]/256.f;

    yl->s4 = yb[4];
    yl->s5 = yb[5]/256.f;

    yl->s6 = yb[6];
    yl->s7 = yb[7]/256.f;

    yl->s8 = yb[16]/16.f;
    yl->s9 = yb[17]/4096.f;

    yl->sa = yb[18]/16.f;
    yl->sb = yb[19]/4096.f;

    yl->sc = yb[20]/16.f;
    yl->sd = yb[21]/4096.f;

    yl->se = yb[22]/16.f;
    yl->sf = yb[23]/4096.f;
}

#ifdef cl_intel_required_subgroup_size
REQD_SUBGROUP_SIZE_16
#elif defined(cl_qcom_reqd_sub_group_size)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_dense_q4_0_matvec_8x(
        global const ushort * q,
        global const half   * d,
        global const float  * y,
        global float        * dst,
        int                   hidden_dim,
        int                   rows) {
    const int nb = hidden_dim/QK4_0;
    const int first_row = get_group_id(0) * Q4_N_DST;
    const int lid = get_sub_group_local_id();

    const int ix = lid/2;
    const int il = 8*(lid%2);
    float16 yl;
    float sumy;

    float8 sumf = 0.f;
    for (int ib = ix; ib < nb; ib += Q4_SIMD_HALF_WIDTH) {
        load_hidden_tile(y + ib*QK4_0 + il, &yl, &sumy);

        sumf.s0 += first_row + 0 < rows ? block_q4_0_dot_transposed(q, d, rows, first_row + 0, ib, sumy, yl, il) : 0.f;
        sumf.s1 += first_row + 1 < rows ? block_q4_0_dot_transposed(q, d, rows, first_row + 1, ib, sumy, yl, il) : 0.f;
        sumf.s2 += first_row + 2 < rows ? block_q4_0_dot_transposed(q, d, rows, first_row + 2, ib, sumy, yl, il) : 0.f;
        sumf.s3 += first_row + 3 < rows ? block_q4_0_dot_transposed(q, d, rows, first_row + 3, ib, sumy, yl, il) : 0.f;
        sumf.s4 += first_row + 4 < rows ? block_q4_0_dot_transposed(q, d, rows, first_row + 4, ib, sumy, yl, il) : 0.f;
        sumf.s5 += first_row + 5 < rows ? block_q4_0_dot_transposed(q, d, rows, first_row + 5, ib, sumy, yl, il) : 0.f;
        sumf.s6 += first_row + 6 < rows ? block_q4_0_dot_transposed(q, d, rows, first_row + 6, ib, sumy, yl, il) : 0.f;
        sumf.s7 += first_row + 7 < rows ? block_q4_0_dot_transposed(q, d, rows, first_row + 7, ib, sumy, yl, il) : 0.f;

    }

    float8 total = (float8)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
        sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
        sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7));

    if (lid == 0) {
        if (first_row + 0 < rows) dst[first_row + 0] = total.s0;
        if (first_row + 1 < rows) dst[first_row + 1] = total.s1;
        if (first_row + 2 < rows) dst[first_row + 2] = total.s2;
        if (first_row + 3 < rows) dst[first_row + 3] = total.s3;
        if (first_row + 4 < rows) dst[first_row + 4] = total.s4;
        if (first_row + 5 < rows) dst[first_row + 5] = total.s5;
        if (first_row + 6 < rows) dst[first_row + 6] = total.s6;
        if (first_row + 7 < rows) dst[first_row + 7] = total.s7;
    }
}

#ifdef cl_intel_required_subgroup_size
REQD_SUBGROUP_SIZE_16
#elif defined(cl_qcom_reqd_sub_group_size)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_indexed_q4_0_matvec_8x(
        global const ushort * q,
        global const half   * d,
        global const float  * y,
        global const int    * ids,
        global float        * dst,
        int                   hidden_dim,
        int                   src_rows,
        int                   out_rows) {
    const int nb = hidden_dim/QK4_0;
    const int first_out = (get_group_id(0) * get_num_sub_groups() + get_sub_group_id()) * INDEXED_ROWS_PER_SG;
    const int lid = get_sub_group_local_id();

    const int row0 = INDEXED_ROWS_PER_SG > 0 && first_out + 0 < out_rows ? ids[first_out + 0] : 0;
    const int row1 = INDEXED_ROWS_PER_SG > 1 && first_out + 1 < out_rows ? ids[first_out + 1] : 0;
    const int row2 = INDEXED_ROWS_PER_SG > 2 && first_out + 2 < out_rows ? ids[first_out + 2] : 0;
    const int row3 = INDEXED_ROWS_PER_SG > 3 && first_out + 3 < out_rows ? ids[first_out + 3] : 0;
    const int row4 = INDEXED_ROWS_PER_SG > 4 && first_out + 4 < out_rows ? ids[first_out + 4] : 0;
    const int row5 = INDEXED_ROWS_PER_SG > 5 && first_out + 5 < out_rows ? ids[first_out + 5] : 0;
    const int row6 = INDEXED_ROWS_PER_SG > 6 && first_out + 6 < out_rows ? ids[first_out + 6] : 0;
    const int row7 = INDEXED_ROWS_PER_SG > 7 && first_out + 7 < out_rows ? ids[first_out + 7] : 0;

    const int ix = lid/2;
    const int il = 8*(lid%2);
    float16 yl;
    float sumy;

    float8 sumf = 0.f;
    for (int ib = ix; ib < nb; ib += Q4_SIMD_HALF_WIDTH) {
        load_hidden_tile(y + ib*QK4_0 + il, &yl, &sumy);

        sumf.s0 += INDEXED_ROWS_PER_SG > 0 && first_out + 0 < out_rows ? block_q4_0_dot_transposed(q, d, src_rows, row0, ib, sumy, yl, il) : 0.f;
        sumf.s1 += INDEXED_ROWS_PER_SG > 1 && first_out + 1 < out_rows ? block_q4_0_dot_transposed(q, d, src_rows, row1, ib, sumy, yl, il) : 0.f;
        sumf.s2 += INDEXED_ROWS_PER_SG > 2 && first_out + 2 < out_rows ? block_q4_0_dot_transposed(q, d, src_rows, row2, ib, sumy, yl, il) : 0.f;
        sumf.s3 += INDEXED_ROWS_PER_SG > 3 && first_out + 3 < out_rows ? block_q4_0_dot_transposed(q, d, src_rows, row3, ib, sumy, yl, il) : 0.f;
        sumf.s4 += INDEXED_ROWS_PER_SG > 4 && first_out + 4 < out_rows ? block_q4_0_dot_transposed(q, d, src_rows, row4, ib, sumy, yl, il) : 0.f;
        sumf.s5 += INDEXED_ROWS_PER_SG > 5 && first_out + 5 < out_rows ? block_q4_0_dot_transposed(q, d, src_rows, row5, ib, sumy, yl, il) : 0.f;
        sumf.s6 += INDEXED_ROWS_PER_SG > 6 && first_out + 6 < out_rows ? block_q4_0_dot_transposed(q, d, src_rows, row6, ib, sumy, yl, il) : 0.f;
        sumf.s7 += INDEXED_ROWS_PER_SG > 7 && first_out + 7 < out_rows ? block_q4_0_dot_transposed(q, d, src_rows, row7, ib, sumy, yl, il) : 0.f;

    }

    float8 total = (float8)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
        sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
        sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7));

    if (lid == 0) {
        if (INDEXED_ROWS_PER_SG > 0 && first_out + 0 < out_rows) dst[first_out + 0] = total.s0;
        if (INDEXED_ROWS_PER_SG > 1 && first_out + 1 < out_rows) dst[first_out + 1] = total.s1;
        if (INDEXED_ROWS_PER_SG > 2 && first_out + 2 < out_rows) dst[first_out + 2] = total.s2;
        if (INDEXED_ROWS_PER_SG > 3 && first_out + 3 < out_rows) dst[first_out + 3] = total.s3;
        if (INDEXED_ROWS_PER_SG > 4 && first_out + 4 < out_rows) dst[first_out + 4] = total.s4;
        if (INDEXED_ROWS_PER_SG > 5 && first_out + 5 < out_rows) dst[first_out + 5] = total.s5;
        if (INDEXED_ROWS_PER_SG > 6 && first_out + 6 < out_rows) dst[first_out + 6] = total.s6;
        if (INDEXED_ROWS_PER_SG > 7 && first_out + 7 < out_rows) dst[first_out + 7] = total.s7;
    }
}

#define INDEXED_AB_BI_WGS 128

#ifdef INDEXED_AB_BI_ASSUME_N8
#define INDEXED_AB_BI_IF_HI_N
#define INDEXED_AB_BI_STORE_HI(row_idx, acc_value, out_ptr) vstore4(convert_float4(acc_value), 0, (out_ptr) + (row_idx)*out_rows)
#else
#define INDEXED_AB_BI_IF_HI_N if (compute_hi_n)
#define INDEXED_AB_BI_STORE_HI(row_idx, acc_value, out_ptr) if (out_b_idx + (row_idx) < n_rows) vstore4(convert_float4(acc_value), 0, (out_ptr) + (row_idx)*out_rows)
#endif

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_8x4(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lid_k = get_local_id(2);
    const int lsz_m = get_local_size(1);
    const int lsz_k = get_local_size(2);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 3;
    const int out4 = (gid_m * lsz_m + lid_m) << 2;
    const int out_off = out4 + out_b_idx * out_rows;

    const int row0 = out4 + 0 < out_rows ? ids[out4 + 0] : 0;
    const int row1 = out4 + 1 < out_rows ? ids[out4 + 1] : 0;
    const int row2 = out4 + 2 < out_rows ? ids[out4 + 2] : 0;
    const int row3 = out4 + 3 < out_rows ? ids[out4 + 3] : 0;

    const int b_row0_pix = out_b_idx * k4_count;
#ifndef INDEXED_AB_BI_ASSUME_N8
    const int compute_hi_n = n_rows > out_b_idx + 4;
#endif

    half4 acc0 = (half4)0, acc1 = (half4)0, acc2 = (half4)0, acc3 = (half4)0;
    half4 acc4 = (half4)0, acc5 = (half4)0, acc6 = (half4)0, acc7 = (half4)0;

    #define INDEXED_AB_BI_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half4 sc4 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort4 bits4 = (ushort4)( \
            q[q_base + row0], \
            q[q_base + row1], \
            q[q_base + row2], \
            q[q_base + row3]); \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        half4 in4, in5, in6, in7; \
        INDEXED_AB_BI_IF_HI_N { \
            in4 = read_imageh(src1, p + 4*k4_count); \
            in5 = read_imageh(src1, p + 5*k4_count); \
            in6 = read_imageh(src1, p + 6*k4_count); \
            in7 = read_imageh(src1, p + 7*k4_count); \
        } \
        half4 w; \
        w.s0 = ((bits4.s0 & 0x000F) - 8) * sc4.s0; \
        w.s1 = ((bits4.s1 & 0x000F) - 8) * sc4.s1; \
        w.s2 = ((bits4.s2 & 0x000F) - 8) * sc4.s2; \
        w.s3 = ((bits4.s3 & 0x000F) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s0) * w; \
        acc1 += (half4)(in1.s0) * w; \
        acc2 += (half4)(in2.s0) * w; \
        acc3 += (half4)(in3.s0) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s0) * w; \
            acc5 += (half4)(in5.s0) * w; \
            acc6 += (half4)(in6.s0) * w; \
            acc7 += (half4)(in7.s0) * w; \
        } \
        w.s0 = (((bits4.s0 & 0x00F0) >> 4) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x00F0) >> 4) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x00F0) >> 4) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x00F0) >> 4) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s1) * w; \
        acc1 += (half4)(in1.s1) * w; \
        acc2 += (half4)(in2.s1) * w; \
        acc3 += (half4)(in3.s1) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s1) * w; \
            acc5 += (half4)(in5.s1) * w; \
            acc6 += (half4)(in6.s1) * w; \
            acc7 += (half4)(in7.s1) * w; \
        } \
        w.s0 = (((bits4.s0 & 0x0F00) >> 8) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x0F00) >> 8) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x0F00) >> 8) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x0F00) >> 8) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s2) * w; \
        acc1 += (half4)(in1.s2) * w; \
        acc2 += (half4)(in2.s2) * w; \
        acc3 += (half4)(in3.s2) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s2) * w; \
            acc5 += (half4)(in5.s2) * w; \
            acc6 += (half4)(in6.s2) * w; \
            acc7 += (half4)(in7.s2) * w; \
        } \
        w.s0 = (((bits4.s0 & 0xF000) >> 12) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0xF000) >> 12) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0xF000) >> 12) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0xF000) >> 12) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s3) * w; \
        acc1 += (half4)(in1.s3) * w; \
        acc2 += (half4)(in2.s3) * w; \
        acc3 += (half4)(in3.s3) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s3) * w; \
            acc5 += (half4)(in5.s3) * w; \
            acc6 += (half4)(in6.s3) * w; \
            acc7 += (half4)(in7.s3) * w; \
        } \
    } while (0)

    if (lsz_k <= 8) {
        const int kb_count = hidden_dim >> 5;
        for (int kb = 0; kb < kb_count; ++kb) {
            const ulong d_base = (ulong) kb * (ulong) src_rows;
            const half4 sc = (half4)(d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3]);
            for (int kk = lid_k; kk < 8; kk += lsz_k) {
                INDEXED_AB_BI_ACCUM_K4((kb << 3) + kk, sc);
            }
        }
    } else {
        for (int k4 = lid_k; k4 < k4_count; k4 += lsz_k) {
            const ulong d_base = (ulong) (k4 >> 3) * (ulong) src_rows;
            const half4 sc = (half4)(d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3]);
            INDEXED_AB_BI_ACCUM_K4(k4, sc);
        }
    }

    #undef INDEXED_AB_BI_ACCUM_K4

    __local half4 sum0[INDEXED_AB_BI_WGS], sum1[INDEXED_AB_BI_WGS], sum2[INDEXED_AB_BI_WGS], sum3[INDEXED_AB_BI_WGS];
    __local half4 sum4[INDEXED_AB_BI_WGS], sum5[INDEXED_AB_BI_WGS], sum6[INDEXED_AB_BI_WGS], sum7[INDEXED_AB_BI_WGS];

    if (lsz_k == 1) {
        __global float * outp = dst + out_off;
        if (out4 + 3 < out_rows) {
            if (out_b_idx + 0 < n_rows) vstore4(convert_float4(acc0), 0, outp + 0*out_rows);
            if (out_b_idx + 1 < n_rows) vstore4(convert_float4(acc1), 0, outp + 1*out_rows);
            if (out_b_idx + 2 < n_rows) vstore4(convert_float4(acc2), 0, outp + 2*out_rows);
            if (out_b_idx + 3 < n_rows) vstore4(convert_float4(acc3), 0, outp + 3*out_rows);
            INDEXED_AB_BI_STORE_HI(4, acc4, outp);
            INDEXED_AB_BI_STORE_HI(5, acc5, outp);
            INDEXED_AB_BI_STORE_HI(6, acc6, outp);
            INDEXED_AB_BI_STORE_HI(7, acc7, outp);
        }
    } else {
        const int slot = lid_m * lsz_k + lid_k;
        sum0[slot] = acc0; sum1[slot] = acc1; sum2[slot] = acc2; sum3[slot] = acc3;
        sum4[slot] = acc4; sum5[slot] = acc5; sum6[slot] = acc6; sum7[slot] = acc7;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = lsz_k >> 1; stride > 0; stride >>= 1) {
            if (lid_k < stride) {
                const int my_slot = lid_m * lsz_k + lid_k;
                const int partner_slot = lid_m * lsz_k + (lid_k + stride);
                sum0[my_slot] += sum0[partner_slot];
                sum1[my_slot] += sum1[partner_slot];
                sum2[my_slot] += sum2[partner_slot];
                sum3[my_slot] += sum3[partner_slot];
                sum4[my_slot] += sum4[partner_slot];
                sum5[my_slot] += sum5[partner_slot];
                sum6[my_slot] += sum6[partner_slot];
                sum7[my_slot] += sum7[partner_slot];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid_k == 0 && out4 + 3 < out_rows) {
            const int final_slot = lid_m * lsz_k;
            __global float * outp = dst + out_off;
            if (out_b_idx + 0 < n_rows) vstore4(convert_float4(sum0[final_slot]), 0, outp + 0*out_rows);
            if (out_b_idx + 1 < n_rows) vstore4(convert_float4(sum1[final_slot]), 0, outp + 1*out_rows);
            if (out_b_idx + 2 < n_rows) vstore4(convert_float4(sum2[final_slot]), 0, outp + 2*out_rows);
            if (out_b_idx + 3 < n_rows) vstore4(convert_float4(sum3[final_slot]), 0, outp + 3*out_rows);
            INDEXED_AB_BI_STORE_HI(4, sum4[final_slot], outp);
            INDEXED_AB_BI_STORE_HI(5, sum5[final_slot], outp);
            INDEXED_AB_BI_STORE_HI(6, sum6[final_slot], outp);
            INDEXED_AB_BI_STORE_HI(7, sum7[final_slot], outp);
        }
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_8x4_lbtile(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lsz_m = get_local_size(1);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 3;
    const int out4 = (gid_m * lsz_m + lid_m) << 2;
    const int out_off = out4 + out_b_idx * out_rows;

    const int row0 = out4 + 0 < out_rows ? ids[out4 + 0] : 0;
    const int row1 = out4 + 1 < out_rows ? ids[out4 + 1] : 0;
    const int row2 = out4 + 2 < out_rows ? ids[out4 + 2] : 0;
    const int row3 = out4 + 3 < out_rows ? ids[out4 + 3] : 0;

    const int b_row0_pix = out_b_idx * k4_count;
#ifndef INDEXED_AB_BI_ASSUME_N8
    const int compute_hi_n = n_rows > out_b_idx + 4;
#endif

    __local half4 b_tile[64];
    half4 acc0 = (half4)0, acc1 = (half4)0, acc2 = (half4)0, acc3 = (half4)0;
    half4 acc4 = (half4)0, acc5 = (half4)0, acc6 = (half4)0, acc7 = (half4)0;

    #define INDEXED_AB_BI_LBTILE_ACCUM_K4(kk_value, sc_value) do { \
        const int kk_idx = (kk_value); \
        const half4 sc4 = (sc_value); \
        const int k4_idx = k4_base + kk_idx; \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort4 bits4 = (ushort4)( \
            q[q_base + row0], \
            q[q_base + row1], \
            q[q_base + row2], \
            q[q_base + row3]); \
        const half4 in0 = b_tile[0*8 + kk_idx]; \
        const half4 in1 = b_tile[1*8 + kk_idx]; \
        const half4 in2 = b_tile[2*8 + kk_idx]; \
        const half4 in3 = b_tile[3*8 + kk_idx]; \
        half4 in4, in5, in6, in7; \
        INDEXED_AB_BI_IF_HI_N { \
            in4 = b_tile[4*8 + kk_idx]; \
            in5 = b_tile[5*8 + kk_idx]; \
            in6 = b_tile[6*8 + kk_idx]; \
            in7 = b_tile[7*8 + kk_idx]; \
        } \
        half4 w; \
        w.s0 = ((bits4.s0 & 0x000F) - 8) * sc4.s0; \
        w.s1 = ((bits4.s1 & 0x000F) - 8) * sc4.s1; \
        w.s2 = ((bits4.s2 & 0x000F) - 8) * sc4.s2; \
        w.s3 = ((bits4.s3 & 0x000F) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s0) * w; \
        acc1 += (half4)(in1.s0) * w; \
        acc2 += (half4)(in2.s0) * w; \
        acc3 += (half4)(in3.s0) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s0) * w; \
            acc5 += (half4)(in5.s0) * w; \
            acc6 += (half4)(in6.s0) * w; \
            acc7 += (half4)(in7.s0) * w; \
        } \
        w.s0 = (((bits4.s0 & 0x00F0) >> 4) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x00F0) >> 4) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x00F0) >> 4) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x00F0) >> 4) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s1) * w; \
        acc1 += (half4)(in1.s1) * w; \
        acc2 += (half4)(in2.s1) * w; \
        acc3 += (half4)(in3.s1) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s1) * w; \
            acc5 += (half4)(in5.s1) * w; \
            acc6 += (half4)(in6.s1) * w; \
            acc7 += (half4)(in7.s1) * w; \
        } \
        w.s0 = (((bits4.s0 & 0x0F00) >> 8) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x0F00) >> 8) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x0F00) >> 8) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x0F00) >> 8) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s2) * w; \
        acc1 += (half4)(in1.s2) * w; \
        acc2 += (half4)(in2.s2) * w; \
        acc3 += (half4)(in3.s2) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s2) * w; \
            acc5 += (half4)(in5.s2) * w; \
            acc6 += (half4)(in6.s2) * w; \
            acc7 += (half4)(in7.s2) * w; \
        } \
        w.s0 = (((bits4.s0 & 0xF000) >> 12) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0xF000) >> 12) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0xF000) >> 12) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0xF000) >> 12) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s3) * w; \
        acc1 += (half4)(in1.s3) * w; \
        acc2 += (half4)(in2.s3) * w; \
        acc3 += (half4)(in3.s3) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s3) * w; \
            acc5 += (half4)(in5.s3) * w; \
            acc6 += (half4)(in6.s3) * w; \
            acc7 += (half4)(in7.s3) * w; \
        } \
    } while (0)

    const int kb_count = hidden_dim >> 5;
    for (int kb = 0; kb < kb_count; ++kb) {
        const int k4_base = kb << 3;
        for (int i = lid_m; i < 64; i += lsz_m) {
            const int nr = i >> 3;
            const int kk = i & 7;
            b_tile[i] = read_imageh(src1, b_row0_pix + nr*k4_count + k4_base + kk);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const ulong d_base = (ulong) kb * (ulong) src_rows;
        const half4 sc = (half4)(d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3]);
        for (int kk = 0; kk < 8; ++kk) {
            INDEXED_AB_BI_LBTILE_ACCUM_K4(kk, sc);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #undef INDEXED_AB_BI_LBTILE_ACCUM_K4

    if (out4 + 3 < out_rows) {
        __global float * outp = dst + out_off;
        if (out_b_idx + 0 < n_rows) vstore4(convert_float4(acc0), 0, outp + 0*out_rows);
        if (out_b_idx + 1 < n_rows) vstore4(convert_float4(acc1), 0, outp + 1*out_rows);
        if (out_b_idx + 2 < n_rows) vstore4(convert_float4(acc2), 0, outp + 2*out_rows);
        if (out_b_idx + 3 < n_rows) vstore4(convert_float4(acc3), 0, outp + 3*out_rows);
        INDEXED_AB_BI_STORE_HI(4, acc4, outp);
        INDEXED_AB_BI_STORE_HI(5, acc5, outp);
        INDEXED_AB_BI_STORE_HI(6, acc6, outp);
        INDEXED_AB_BI_STORE_HI(7, acc7, outp);
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_8x4_nosplit(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lsz_m = get_local_size(1);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 3;
    const int out4 = (gid_m * lsz_m + lid_m) << 2;
    const int out_off = out4 + out_b_idx * out_rows;
#ifndef INDEXED_AB_BI_ASSUME_N8
    const int compute_hi_n = n_rows > out_b_idx + 4;
#endif

    const int row0 = out4 + 0 < out_rows ? ids[out4 + 0] : 0;
    const int row1 = out4 + 1 < out_rows ? ids[out4 + 1] : 0;
    const int row2 = out4 + 2 < out_rows ? ids[out4 + 2] : 0;
    const int row3 = out4 + 3 < out_rows ? ids[out4 + 3] : 0;

    const int b_row0_pix = out_b_idx * k4_count;

    half4 acc0 = (half4)0, acc1 = (half4)0, acc2 = (half4)0, acc3 = (half4)0;
    half4 acc4 = (half4)0, acc5 = (half4)0, acc6 = (half4)0, acc7 = (half4)0;

    #define INDEXED_AB_BI_NOSPLIT_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half4 sc4 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort4 bits4 = (ushort4)( \
            q[q_base + row0], \
            q[q_base + row1], \
            q[q_base + row2], \
            q[q_base + row3]); \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        half4 in4, in5, in6, in7; \
        INDEXED_AB_BI_IF_HI_N { \
            in4 = read_imageh(src1, p + 4*k4_count); \
            in5 = read_imageh(src1, p + 5*k4_count); \
            in6 = read_imageh(src1, p + 6*k4_count); \
            in7 = read_imageh(src1, p + 7*k4_count); \
        } \
        half4 w; \
        w.s0 = ((bits4.s0 & 0x000F) - 8) * sc4.s0; \
        w.s1 = ((bits4.s1 & 0x000F) - 8) * sc4.s1; \
        w.s2 = ((bits4.s2 & 0x000F) - 8) * sc4.s2; \
        w.s3 = ((bits4.s3 & 0x000F) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s0) * w; \
        acc1 += (half4)(in1.s0) * w; \
        acc2 += (half4)(in2.s0) * w; \
        acc3 += (half4)(in3.s0) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s0) * w; \
            acc5 += (half4)(in5.s0) * w; \
            acc6 += (half4)(in6.s0) * w; \
            acc7 += (half4)(in7.s0) * w; \
        } \
        w.s0 = (((bits4.s0 & 0x00F0) >> 4) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x00F0) >> 4) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x00F0) >> 4) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x00F0) >> 4) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s1) * w; \
        acc1 += (half4)(in1.s1) * w; \
        acc2 += (half4)(in2.s1) * w; \
        acc3 += (half4)(in3.s1) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s1) * w; \
            acc5 += (half4)(in5.s1) * w; \
            acc6 += (half4)(in6.s1) * w; \
            acc7 += (half4)(in7.s1) * w; \
        } \
        w.s0 = (((bits4.s0 & 0x0F00) >> 8) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x0F00) >> 8) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x0F00) >> 8) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x0F00) >> 8) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s2) * w; \
        acc1 += (half4)(in1.s2) * w; \
        acc2 += (half4)(in2.s2) * w; \
        acc3 += (half4)(in3.s2) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s2) * w; \
            acc5 += (half4)(in5.s2) * w; \
            acc6 += (half4)(in6.s2) * w; \
            acc7 += (half4)(in7.s2) * w; \
        } \
        w.s0 = (((bits4.s0 & 0xF000) >> 12) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0xF000) >> 12) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0xF000) >> 12) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0xF000) >> 12) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s3) * w; \
        acc1 += (half4)(in1.s3) * w; \
        acc2 += (half4)(in2.s3) * w; \
        acc3 += (half4)(in3.s3) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s3) * w; \
            acc5 += (half4)(in5.s3) * w; \
            acc6 += (half4)(in6.s3) * w; \
            acc7 += (half4)(in7.s3) * w; \
        } \
    } while (0)

    const int kb_count = hidden_dim >> 5;
    for (int kb = 0; kb < kb_count; ++kb) {
        const ulong d_base = (ulong) kb * (ulong) src_rows;
        const half4 sc = (half4)(d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3]);
        const int k4_base = kb << 3;
        for (int kk = 0; kk < 8; ++kk) {
            INDEXED_AB_BI_NOSPLIT_ACCUM_K4(k4_base + kk, sc);
        }
    }

    #undef INDEXED_AB_BI_NOSPLIT_ACCUM_K4

    if (out4 + 3 < out_rows) {
        __global float * outp = dst + out_off;
        if (out_b_idx + 0 < n_rows) vstore4(convert_float4(acc0), 0, outp + 0*out_rows);
        if (out_b_idx + 1 < n_rows) vstore4(convert_float4(acc1), 0, outp + 1*out_rows);
        if (out_b_idx + 2 < n_rows) vstore4(convert_float4(acc2), 0, outp + 2*out_rows);
        if (out_b_idx + 3 < n_rows) vstore4(convert_float4(acc3), 0, outp + 3*out_rows);
        INDEXED_AB_BI_STORE_HI(4, acc4, outp);
        INDEXED_AB_BI_STORE_HI(5, acc5, outp);
        INDEXED_AB_BI_STORE_HI(6, acc6, outp);
        INDEXED_AB_BI_STORE_HI(7, acc7, outp);
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_8x4_nosplit_prefetch(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lsz_m = get_local_size(1);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 3;
    const int out4 = (gid_m * lsz_m + lid_m) << 2;
    const int out_off = out4 + out_b_idx * out_rows;
#ifndef INDEXED_AB_BI_ASSUME_N8
    const int compute_hi_n = n_rows > out_b_idx + 4;
#endif

    const int row0 = out4 + 0 < out_rows ? ids[out4 + 0] : 0;
    const int row1 = out4 + 1 < out_rows ? ids[out4 + 1] : 0;
    const int row2 = out4 + 2 < out_rows ? ids[out4 + 2] : 0;
    const int row3 = out4 + 3 < out_rows ? ids[out4 + 3] : 0;

    const int b_row0_pix = out_b_idx * k4_count;

    half4 acc0 = (half4)0, acc1 = (half4)0, acc2 = (half4)0, acc3 = (half4)0;
    half4 acc4 = (half4)0, acc5 = (half4)0, acc6 = (half4)0, acc7 = (half4)0;

    #define INDEXED_AB_BI_PREFETCH_Q4(k4_value) do { \
        const ulong q_base_pf = (ulong) (k4_value) * (ulong) src_rows; \
        prefetch(q + q_base_pf + row0, 1); \
        prefetch(q + q_base_pf + row1, 1); \
        prefetch(q + q_base_pf + row2, 1); \
        prefetch(q + q_base_pf + row3, 1); \
    } while (0)

    #define INDEXED_AB_BI_PREFETCH_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half4 sc4 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort4 bits4 = (ushort4)( \
            q[q_base + row0], \
            q[q_base + row1], \
            q[q_base + row2], \
            q[q_base + row3]); \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        half4 in4, in5, in6, in7; \
        INDEXED_AB_BI_IF_HI_N { \
            in4 = read_imageh(src1, p + 4*k4_count); \
            in5 = read_imageh(src1, p + 5*k4_count); \
            in6 = read_imageh(src1, p + 6*k4_count); \
            in7 = read_imageh(src1, p + 7*k4_count); \
        } \
        half4 w; \
        w.s0 = ((bits4.s0 & 0x000F) - 8) * sc4.s0; \
        w.s1 = ((bits4.s1 & 0x000F) - 8) * sc4.s1; \
        w.s2 = ((bits4.s2 & 0x000F) - 8) * sc4.s2; \
        w.s3 = ((bits4.s3 & 0x000F) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s0) * w; \
        acc1 += (half4)(in1.s0) * w; \
        acc2 += (half4)(in2.s0) * w; \
        acc3 += (half4)(in3.s0) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s0) * w; \
            acc5 += (half4)(in5.s0) * w; \
            acc6 += (half4)(in6.s0) * w; \
            acc7 += (half4)(in7.s0) * w; \
        } \
        w.s0 = (((bits4.s0 & 0x00F0) >> 4) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x00F0) >> 4) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x00F0) >> 4) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x00F0) >> 4) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s1) * w; \
        acc1 += (half4)(in1.s1) * w; \
        acc2 += (half4)(in2.s1) * w; \
        acc3 += (half4)(in3.s1) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s1) * w; \
            acc5 += (half4)(in5.s1) * w; \
            acc6 += (half4)(in6.s1) * w; \
            acc7 += (half4)(in7.s1) * w; \
        } \
        w.s0 = (((bits4.s0 & 0x0F00) >> 8) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x0F00) >> 8) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x0F00) >> 8) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x0F00) >> 8) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s2) * w; \
        acc1 += (half4)(in1.s2) * w; \
        acc2 += (half4)(in2.s2) * w; \
        acc3 += (half4)(in3.s2) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s2) * w; \
            acc5 += (half4)(in5.s2) * w; \
            acc6 += (half4)(in6.s2) * w; \
            acc7 += (half4)(in7.s2) * w; \
        } \
        w.s0 = (((bits4.s0 & 0xF000) >> 12) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0xF000) >> 12) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0xF000) >> 12) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0xF000) >> 12) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s3) * w; \
        acc1 += (half4)(in1.s3) * w; \
        acc2 += (half4)(in2.s3) * w; \
        acc3 += (half4)(in3.s3) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half4)(in4.s3) * w; \
            acc5 += (half4)(in5.s3) * w; \
            acc6 += (half4)(in6.s3) * w; \
            acc7 += (half4)(in7.s3) * w; \
        } \
    } while (0)

    const int kb_count = hidden_dim >> 5;
    if (k4_count > 0) {
        INDEXED_AB_BI_PREFETCH_Q4(0);
    }
    if (k4_count > 1) {
        INDEXED_AB_BI_PREFETCH_Q4(1);
    }
    for (int kb = 0; kb < kb_count; ++kb) {
        const ulong d_base = (ulong) kb * (ulong) src_rows;
        const half4 sc = (half4)(d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3]);
        const int k4_base = kb << 3;
        for (int kk = 0; kk < 8; ++kk) {
            const int pf_k4 = k4_base + kk + 2;
            if (pf_k4 < k4_count) {
                INDEXED_AB_BI_PREFETCH_Q4(pf_k4);
            }
            INDEXED_AB_BI_PREFETCH_ACCUM_K4(k4_base + kk, sc);
        }
    }

    #undef INDEXED_AB_BI_PREFETCH_ACCUM_K4
    #undef INDEXED_AB_BI_PREFETCH_Q4

    if (out4 + 3 < out_rows) {
        __global float * outp = dst + out_off;
        if (out_b_idx + 0 < n_rows) vstore4(convert_float4(acc0), 0, outp + 0*out_rows);
        if (out_b_idx + 1 < n_rows) vstore4(convert_float4(acc1), 0, outp + 1*out_rows);
        if (out_b_idx + 2 < n_rows) vstore4(convert_float4(acc2), 0, outp + 2*out_rows);
        if (out_b_idx + 3 < n_rows) vstore4(convert_float4(acc3), 0, outp + 3*out_rows);
        INDEXED_AB_BI_STORE_HI(4, acc4, outp);
        INDEXED_AB_BI_STORE_HI(5, acc5, outp);
        INDEXED_AB_BI_STORE_HI(6, acc6, outp);
        INDEXED_AB_BI_STORE_HI(7, acc7, outp);
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_4x4(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lid_k = get_local_id(2);
    const int lsz_m = get_local_size(1);
    const int lsz_k = get_local_size(2);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 2;
    const int out4 = (gid_m * lsz_m + lid_m) << 2;
    const int out_off = out4 + out_b_idx * out_rows;

    const int row0 = out4 + 0 < out_rows ? ids[out4 + 0] : 0;
    const int row1 = out4 + 1 < out_rows ? ids[out4 + 1] : 0;
    const int row2 = out4 + 2 < out_rows ? ids[out4 + 2] : 0;
    const int row3 = out4 + 3 < out_rows ? ids[out4 + 3] : 0;

    const int b_row0_pix = out_b_idx * k4_count;

    half4 acc0 = (half4)0, acc1 = (half4)0, acc2 = (half4)0, acc3 = (half4)0;

    #define INDEXED_AB_BI4_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half4 sc4 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort4 bits4 = (ushort4)( \
            q[q_base + row0], \
            q[q_base + row1], \
            q[q_base + row2], \
            q[q_base + row3]); \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        half4 w; \
        w.s0 = ((bits4.s0 & 0x000F) - 8) * sc4.s0; \
        w.s1 = ((bits4.s1 & 0x000F) - 8) * sc4.s1; \
        w.s2 = ((bits4.s2 & 0x000F) - 8) * sc4.s2; \
        w.s3 = ((bits4.s3 & 0x000F) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s0) * w; \
        acc1 += (half4)(in1.s0) * w; \
        acc2 += (half4)(in2.s0) * w; \
        acc3 += (half4)(in3.s0) * w; \
        w.s0 = (((bits4.s0 & 0x00F0) >> 4) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x00F0) >> 4) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x00F0) >> 4) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x00F0) >> 4) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s1) * w; \
        acc1 += (half4)(in1.s1) * w; \
        acc2 += (half4)(in2.s1) * w; \
        acc3 += (half4)(in3.s1) * w; \
        w.s0 = (((bits4.s0 & 0x0F00) >> 8) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x0F00) >> 8) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x0F00) >> 8) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x0F00) >> 8) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s2) * w; \
        acc1 += (half4)(in1.s2) * w; \
        acc2 += (half4)(in2.s2) * w; \
        acc3 += (half4)(in3.s2) * w; \
        w.s0 = (((bits4.s0 & 0xF000) >> 12) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0xF000) >> 12) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0xF000) >> 12) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0xF000) >> 12) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s3) * w; \
        acc1 += (half4)(in1.s3) * w; \
        acc2 += (half4)(in2.s3) * w; \
        acc3 += (half4)(in3.s3) * w; \
    } while (0)

    if (lsz_k <= 8) {
        const int kb_count = hidden_dim >> 5;
        for (int kb = 0; kb < kb_count; ++kb) {
            const ulong d_base = (ulong) kb * (ulong) src_rows;
            const half4 sc = (half4)(d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3]);
            for (int kk = lid_k; kk < 8; kk += lsz_k) {
                INDEXED_AB_BI4_ACCUM_K4((kb << 3) + kk, sc);
            }
        }
    } else {
        for (int k4 = lid_k; k4 < k4_count; k4 += lsz_k) {
            const ulong d_base = (ulong) (k4 >> 3) * (ulong) src_rows;
            const half4 sc = (half4)(d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3]);
            INDEXED_AB_BI4_ACCUM_K4(k4, sc);
        }
    }

    #undef INDEXED_AB_BI4_ACCUM_K4

    __local half4 sum0[INDEXED_AB_BI_WGS], sum1[INDEXED_AB_BI_WGS], sum2[INDEXED_AB_BI_WGS], sum3[INDEXED_AB_BI_WGS];

    if (lsz_k == 1) {
        __global float * outp = dst + out_off;
        if (out4 + 3 < out_rows) {
            if (out_b_idx + 0 < n_rows) vstore4(convert_float4(acc0), 0, outp + 0*out_rows);
            if (out_b_idx + 1 < n_rows) vstore4(convert_float4(acc1), 0, outp + 1*out_rows);
            if (out_b_idx + 2 < n_rows) vstore4(convert_float4(acc2), 0, outp + 2*out_rows);
            if (out_b_idx + 3 < n_rows) vstore4(convert_float4(acc3), 0, outp + 3*out_rows);
        }
    } else {
        const int slot = lid_m * lsz_k + lid_k;
        sum0[slot] = acc0; sum1[slot] = acc1; sum2[slot] = acc2; sum3[slot] = acc3;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = lsz_k >> 1; stride > 0; stride >>= 1) {
            if (lid_k < stride) {
                const int my_slot = lid_m * lsz_k + lid_k;
                const int partner_slot = lid_m * lsz_k + (lid_k + stride);
                sum0[my_slot] += sum0[partner_slot];
                sum1[my_slot] += sum1[partner_slot];
                sum2[my_slot] += sum2[partner_slot];
                sum3[my_slot] += sum3[partner_slot];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid_k == 0 && out4 + 3 < out_rows) {
            const int final_slot = lid_m * lsz_k;
            __global float * outp = dst + out_off;
            if (out_b_idx + 0 < n_rows) vstore4(convert_float4(sum0[final_slot]), 0, outp + 0*out_rows);
            if (out_b_idx + 1 < n_rows) vstore4(convert_float4(sum1[final_slot]), 0, outp + 1*out_rows);
            if (out_b_idx + 2 < n_rows) vstore4(convert_float4(sum2[final_slot]), 0, outp + 2*out_rows);
            if (out_b_idx + 3 < n_rows) vstore4(convert_float4(sum3[final_slot]), 0, outp + 3*out_rows);
        }
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_4x4_nosplit(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lsz_m = get_local_size(1);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 2;
    const int out4 = (gid_m * lsz_m + lid_m) << 2;
    const int out_off = out4 + out_b_idx * out_rows;

    const int row0 = out4 + 0 < out_rows ? ids[out4 + 0] : 0;
    const int row1 = out4 + 1 < out_rows ? ids[out4 + 1] : 0;
    const int row2 = out4 + 2 < out_rows ? ids[out4 + 2] : 0;
    const int row3 = out4 + 3 < out_rows ? ids[out4 + 3] : 0;

    const int b_row0_pix = out_b_idx * k4_count;

    half4 acc0 = (half4)0, acc1 = (half4)0, acc2 = (half4)0, acc3 = (half4)0;

    #define INDEXED_AB_BI4_NOSPLIT_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half4 sc4 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort4 bits4 = (ushort4)( \
            q[q_base + row0], \
            q[q_base + row1], \
            q[q_base + row2], \
            q[q_base + row3]); \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        half4 w; \
        w.s0 = ((bits4.s0 & 0x000F) - 8) * sc4.s0; \
        w.s1 = ((bits4.s1 & 0x000F) - 8) * sc4.s1; \
        w.s2 = ((bits4.s2 & 0x000F) - 8) * sc4.s2; \
        w.s3 = ((bits4.s3 & 0x000F) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s0) * w; \
        acc1 += (half4)(in1.s0) * w; \
        acc2 += (half4)(in2.s0) * w; \
        acc3 += (half4)(in3.s0) * w; \
        w.s0 = (((bits4.s0 & 0x00F0) >> 4) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x00F0) >> 4) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x00F0) >> 4) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x00F0) >> 4) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s1) * w; \
        acc1 += (half4)(in1.s1) * w; \
        acc2 += (half4)(in2.s1) * w; \
        acc3 += (half4)(in3.s1) * w; \
        w.s0 = (((bits4.s0 & 0x0F00) >> 8) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0x0F00) >> 8) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0x0F00) >> 8) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0x0F00) >> 8) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s2) * w; \
        acc1 += (half4)(in1.s2) * w; \
        acc2 += (half4)(in2.s2) * w; \
        acc3 += (half4)(in3.s2) * w; \
        w.s0 = (((bits4.s0 & 0xF000) >> 12) - 8) * sc4.s0; \
        w.s1 = (((bits4.s1 & 0xF000) >> 12) - 8) * sc4.s1; \
        w.s2 = (((bits4.s2 & 0xF000) >> 12) - 8) * sc4.s2; \
        w.s3 = (((bits4.s3 & 0xF000) >> 12) - 8) * sc4.s3; \
        acc0 += (half4)(in0.s3) * w; \
        acc1 += (half4)(in1.s3) * w; \
        acc2 += (half4)(in2.s3) * w; \
        acc3 += (half4)(in3.s3) * w; \
    } while (0)

    const int kb_count = hidden_dim >> 5;
    for (int kb = 0; kb < kb_count; ++kb) {
        const ulong d_base = (ulong) kb * (ulong) src_rows;
        const half4 sc = (half4)(d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3]);
        const int k4_base = kb << 3;
        for (int kk = 0; kk < 8; ++kk) {
            INDEXED_AB_BI4_NOSPLIT_ACCUM_K4(k4_base + kk, sc);
        }
    }

    #undef INDEXED_AB_BI4_NOSPLIT_ACCUM_K4

    if (out4 + 3 < out_rows) {
        __global float * outp = dst + out_off;
        if (out_b_idx + 0 < n_rows) vstore4(convert_float4(acc0), 0, outp + 0*out_rows);
        if (out_b_idx + 1 < n_rows) vstore4(convert_float4(acc1), 0, outp + 1*out_rows);
        if (out_b_idx + 2 < n_rows) vstore4(convert_float4(acc2), 0, outp + 2*out_rows);
        if (out_b_idx + 3 < n_rows) vstore4(convert_float4(acc3), 0, outp + 3*out_rows);
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_4x2_nosplit(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lsz_m = get_local_size(1);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 2;
    const int out2 = (gid_m * lsz_m + lid_m) << 1;
    const int out_off = out2 + out_b_idx * out_rows;

    const int row0 = out2 + 0 < out_rows ? ids[out2 + 0] : 0;
    const int row1 = out2 + 1 < out_rows ? ids[out2 + 1] : 0;

    const int b_row0_pix = out_b_idx * k4_count;

    half2 acc0 = (half2)0, acc1 = (half2)0, acc2 = (half2)0, acc3 = (half2)0;

    #define INDEXED_AB_BI4X2_NOSPLIT_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half2 sc2 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort2 bits2 = (ushort2)(q[q_base + row0], q[q_base + row1]); \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        half2 w; \
        w.s0 = ((bits2.s0 & 0x000F) - 8) * sc2.s0; \
        w.s1 = ((bits2.s1 & 0x000F) - 8) * sc2.s1; \
        acc0 += (half2)(in0.s0) * w; \
        acc1 += (half2)(in1.s0) * w; \
        acc2 += (half2)(in2.s0) * w; \
        acc3 += (half2)(in3.s0) * w; \
        w.s0 = (((bits2.s0 & 0x00F0) >> 4) - 8) * sc2.s0; \
        w.s1 = (((bits2.s1 & 0x00F0) >> 4) - 8) * sc2.s1; \
        acc0 += (half2)(in0.s1) * w; \
        acc1 += (half2)(in1.s1) * w; \
        acc2 += (half2)(in2.s1) * w; \
        acc3 += (half2)(in3.s1) * w; \
        w.s0 = (((bits2.s0 & 0x0F00) >> 8) - 8) * sc2.s0; \
        w.s1 = (((bits2.s1 & 0x0F00) >> 8) - 8) * sc2.s1; \
        acc0 += (half2)(in0.s2) * w; \
        acc1 += (half2)(in1.s2) * w; \
        acc2 += (half2)(in2.s2) * w; \
        acc3 += (half2)(in3.s2) * w; \
        w.s0 = (((bits2.s0 & 0xF000) >> 12) - 8) * sc2.s0; \
        w.s1 = (((bits2.s1 & 0xF000) >> 12) - 8) * sc2.s1; \
        acc0 += (half2)(in0.s3) * w; \
        acc1 += (half2)(in1.s3) * w; \
        acc2 += (half2)(in2.s3) * w; \
        acc3 += (half2)(in3.s3) * w; \
    } while (0)

    const int kb_count = hidden_dim >> 5;
    for (int kb = 0; kb < kb_count; ++kb) {
        const ulong d_base = (ulong) kb * (ulong) src_rows;
        const half2 sc = (half2)(d[d_base + row0], d[d_base + row1]);
        const int k4_base = kb << 3;
        for (int kk = 0; kk < 8; ++kk) {
            INDEXED_AB_BI4X2_NOSPLIT_ACCUM_K4(k4_base + kk, sc);
        }
    }

    #undef INDEXED_AB_BI4X2_NOSPLIT_ACCUM_K4

    if (out2 + 1 < out_rows) {
        __global float * outp = dst + out_off;
        if (out_b_idx + 0 < n_rows) vstore2(convert_float2(acc0), 0, outp + 0*out_rows);
        if (out_b_idx + 1 < n_rows) vstore2(convert_float2(acc1), 0, outp + 1*out_rows);
        if (out_b_idx + 2 < n_rows) vstore2(convert_float2(acc2), 0, outp + 2*out_rows);
        if (out_b_idx + 3 < n_rows) vstore2(convert_float2(acc3), 0, outp + 3*out_rows);
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_8x1_nosplit(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lsz_m = get_local_size(1);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 3;
    const int out = gid_m * lsz_m + lid_m;
    const int out_off = out + out_b_idx * out_rows;

    const int row = out < out_rows ? ids[out] : 0;

    const int b_row0_pix = out_b_idx * k4_count;
#ifndef INDEXED_AB_BI_ASSUME_N8
    const int compute_hi_n = n_rows > out_b_idx + 4;
#endif

    half acc0 = (half)0, acc1 = (half)0, acc2 = (half)0, acc3 = (half)0;
    half acc4 = (half)0, acc5 = (half)0, acc6 = (half)0, acc7 = (half)0;

    #define INDEXED_AB_BI1_NOSPLIT_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half sc1 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort bits = q[q_base + row]; \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        half4 in4, in5, in6, in7; \
        INDEXED_AB_BI_IF_HI_N { \
            in4 = read_imageh(src1, p + 4*k4_count); \
            in5 = read_imageh(src1, p + 5*k4_count); \
            in6 = read_imageh(src1, p + 6*k4_count); \
            in7 = read_imageh(src1, p + 7*k4_count); \
        } \
        half w; \
        w = ((bits & 0x000F) - 8) * sc1; \
        acc0 += in0.s0 * w; \
        acc1 += in1.s0 * w; \
        acc2 += in2.s0 * w; \
        acc3 += in3.s0 * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += in4.s0 * w; \
            acc5 += in5.s0 * w; \
            acc6 += in6.s0 * w; \
            acc7 += in7.s0 * w; \
        } \
        w = (((bits & 0x00F0) >> 4) - 8) * sc1; \
        acc0 += in0.s1 * w; \
        acc1 += in1.s1 * w; \
        acc2 += in2.s1 * w; \
        acc3 += in3.s1 * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += in4.s1 * w; \
            acc5 += in5.s1 * w; \
            acc6 += in6.s1 * w; \
            acc7 += in7.s1 * w; \
        } \
        w = (((bits & 0x0F00) >> 8) - 8) * sc1; \
        acc0 += in0.s2 * w; \
        acc1 += in1.s2 * w; \
        acc2 += in2.s2 * w; \
        acc3 += in3.s2 * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += in4.s2 * w; \
            acc5 += in5.s2 * w; \
            acc6 += in6.s2 * w; \
            acc7 += in7.s2 * w; \
        } \
        w = (((bits & 0xF000) >> 12) - 8) * sc1; \
        acc0 += in0.s3 * w; \
        acc1 += in1.s3 * w; \
        acc2 += in2.s3 * w; \
        acc3 += in3.s3 * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += in4.s3 * w; \
            acc5 += in5.s3 * w; \
            acc6 += in6.s3 * w; \
            acc7 += in7.s3 * w; \
        } \
    } while (0)

    const int kb_count = hidden_dim >> 5;
    for (int kb = 0; kb < kb_count; ++kb) {
        const ulong d_base = (ulong) kb * (ulong) src_rows;
        const half sc = d[d_base + row];
        const int k4_base = kb << 3;
        for (int kk = 0; kk < 8; ++kk) {
            INDEXED_AB_BI1_NOSPLIT_ACCUM_K4(k4_base + kk, sc);
        }
    }

    #undef INDEXED_AB_BI1_NOSPLIT_ACCUM_K4

    if (out < out_rows) {
        __global float * outp = dst + out_off;
        if (out_b_idx + 0 < n_rows) outp[0*out_rows] = convert_float(acc0);
        if (out_b_idx + 1 < n_rows) outp[1*out_rows] = convert_float(acc1);
        if (out_b_idx + 2 < n_rows) outp[2*out_rows] = convert_float(acc2);
        if (out_b_idx + 3 < n_rows) outp[3*out_rows] = convert_float(acc3);
        if (out_b_idx + 4 < n_rows) outp[4*out_rows] = convert_float(acc4);
        if (out_b_idx + 5 < n_rows) outp[5*out_rows] = convert_float(acc5);
        if (out_b_idx + 6 < n_rows) outp[6*out_rows] = convert_float(acc6);
        if (out_b_idx + 7 < n_rows) outp[7*out_rows] = convert_float(acc7);
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_8x2_nosplit(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lsz_m = get_local_size(1);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 3;
    const int out2 = (gid_m * lsz_m + lid_m) << 1;
    const int out_off = out2 + out_b_idx * out_rows;

    const int row0 = out2 + 0 < out_rows ? ids[out2 + 0] : 0;
    const int row1 = out2 + 1 < out_rows ? ids[out2 + 1] : 0;

    const int b_row0_pix = out_b_idx * k4_count;
#ifndef INDEXED_AB_BI_ASSUME_N8
    const int compute_hi_n = n_rows > out_b_idx + 4;
#endif

    half2 acc0 = (half2)0, acc1 = (half2)0, acc2 = (half2)0, acc3 = (half2)0;
    half2 acc4 = (half2)0, acc5 = (half2)0, acc6 = (half2)0, acc7 = (half2)0;

    #define INDEXED_AB_BI2_NOSPLIT_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half2 sc2 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort2 bits2 = (ushort2)(q[q_base + row0], q[q_base + row1]); \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        half4 in4, in5, in6, in7; \
        INDEXED_AB_BI_IF_HI_N { \
            in4 = read_imageh(src1, p + 4*k4_count); \
            in5 = read_imageh(src1, p + 5*k4_count); \
            in6 = read_imageh(src1, p + 6*k4_count); \
            in7 = read_imageh(src1, p + 7*k4_count); \
        } \
        half2 w; \
        w.s0 = ((bits2.s0 & 0x000F) - 8) * sc2.s0; \
        w.s1 = ((bits2.s1 & 0x000F) - 8) * sc2.s1; \
        acc0 += (half2)(in0.s0) * w; \
        acc1 += (half2)(in1.s0) * w; \
        acc2 += (half2)(in2.s0) * w; \
        acc3 += (half2)(in3.s0) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half2)(in4.s0) * w; \
            acc5 += (half2)(in5.s0) * w; \
            acc6 += (half2)(in6.s0) * w; \
            acc7 += (half2)(in7.s0) * w; \
        } \
        w.s0 = (((bits2.s0 & 0x00F0) >> 4) - 8) * sc2.s0; \
        w.s1 = (((bits2.s1 & 0x00F0) >> 4) - 8) * sc2.s1; \
        acc0 += (half2)(in0.s1) * w; \
        acc1 += (half2)(in1.s1) * w; \
        acc2 += (half2)(in2.s1) * w; \
        acc3 += (half2)(in3.s1) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half2)(in4.s1) * w; \
            acc5 += (half2)(in5.s1) * w; \
            acc6 += (half2)(in6.s1) * w; \
            acc7 += (half2)(in7.s1) * w; \
        } \
        w.s0 = (((bits2.s0 & 0x0F00) >> 8) - 8) * sc2.s0; \
        w.s1 = (((bits2.s1 & 0x0F00) >> 8) - 8) * sc2.s1; \
        acc0 += (half2)(in0.s2) * w; \
        acc1 += (half2)(in1.s2) * w; \
        acc2 += (half2)(in2.s2) * w; \
        acc3 += (half2)(in3.s2) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half2)(in4.s2) * w; \
            acc5 += (half2)(in5.s2) * w; \
            acc6 += (half2)(in6.s2) * w; \
            acc7 += (half2)(in7.s2) * w; \
        } \
        w.s0 = (((bits2.s0 & 0xF000) >> 12) - 8) * sc2.s0; \
        w.s1 = (((bits2.s1 & 0xF000) >> 12) - 8) * sc2.s1; \
        acc0 += (half2)(in0.s3) * w; \
        acc1 += (half2)(in1.s3) * w; \
        acc2 += (half2)(in2.s3) * w; \
        acc3 += (half2)(in3.s3) * w; \
        INDEXED_AB_BI_IF_HI_N { \
            acc4 += (half2)(in4.s3) * w; \
            acc5 += (half2)(in5.s3) * w; \
            acc6 += (half2)(in6.s3) * w; \
            acc7 += (half2)(in7.s3) * w; \
        } \
    } while (0)

    const int kb_count = hidden_dim >> 5;
    for (int kb = 0; kb < kb_count; ++kb) {
        const ulong d_base = (ulong) kb * (ulong) src_rows;
        const half2 sc = (half2)(d[d_base + row0], d[d_base + row1]);
        const int k4_base = kb << 3;
        for (int kk = 0; kk < 8; ++kk) {
            INDEXED_AB_BI2_NOSPLIT_ACCUM_K4(k4_base + kk, sc);
        }
    }

    #undef INDEXED_AB_BI2_NOSPLIT_ACCUM_K4

    if (out2 + 1 < out_rows) {
        __global float * outp = dst + out_off;
        if (out_b_idx + 0 < n_rows) vstore2(convert_float2(acc0), 0, outp + 0*out_rows);
        if (out_b_idx + 1 < n_rows) vstore2(convert_float2(acc1), 0, outp + 1*out_rows);
        if (out_b_idx + 2 < n_rows) vstore2(convert_float2(acc2), 0, outp + 2*out_rows);
        if (out_b_idx + 3 < n_rows) vstore2(convert_float2(acc3), 0, outp + 3*out_rows);
        if (out_b_idx + 4 < n_rows) vstore2(convert_float2(acc4), 0, outp + 4*out_rows);
        if (out_b_idx + 5 < n_rows) vstore2(convert_float2(acc5), 0, outp + 5*out_rows);
        if (out_b_idx + 6 < n_rows) vstore2(convert_float2(acc6), 0, outp + 6*out_rows);
        if (out_b_idx + 7 < n_rows) vstore2(convert_float2(acc7), 0, outp + 7*out_rows);
    }
}

#ifdef cl_qcom_reqd_sub_group_size
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_indexed_q4_0_matvec_Ab_Bi_8x8_nosplit(
    global const ushort * q,
    global const half   * d,
    __read_only image1d_buffer_t src1,
    global const int    * ids,
    global float        * dst,
    int                   src_rows,
    int                   out_rows,
    int                   hidden_dim,
    int                   n_rows) {
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lsz_m = get_local_size(1);

    const int k4_count = hidden_dim >> 2;
    const int out_b_idx = gid_n << 3;
    const int out8 = (gid_m * lsz_m + lid_m) << 3;
    const int out_off = out8 + out_b_idx * out_rows;

    const int row0 = out8 + 0 < out_rows ? ids[out8 + 0] : 0;
    const int row1 = out8 + 1 < out_rows ? ids[out8 + 1] : 0;
    const int row2 = out8 + 2 < out_rows ? ids[out8 + 2] : 0;
    const int row3 = out8 + 3 < out_rows ? ids[out8 + 3] : 0;
    const int row4 = out8 + 4 < out_rows ? ids[out8 + 4] : 0;
    const int row5 = out8 + 5 < out_rows ? ids[out8 + 5] : 0;
    const int row6 = out8 + 6 < out_rows ? ids[out8 + 6] : 0;
    const int row7 = out8 + 7 < out_rows ? ids[out8 + 7] : 0;

    const int b_row0_pix = out_b_idx * k4_count;

    half8 acc0 = (half8)0, acc1 = (half8)0, acc2 = (half8)0, acc3 = (half8)0;
    half8 acc4 = (half8)0, acc5 = (half8)0, acc6 = (half8)0, acc7 = (half8)0;

    #define INDEXED_AB_BI8_NOSPLIT_ACCUM_K4(k4_value, sc_value) do { \
        const int k4_idx = (k4_value); \
        const half8 sc8 = (sc_value); \
        const ulong q_base = (ulong) k4_idx * (ulong) src_rows; \
        const ushort8 bits8 = (ushort8)( \
            q[q_base + row0], q[q_base + row1], q[q_base + row2], q[q_base + row3], \
            q[q_base + row4], q[q_base + row5], q[q_base + row6], q[q_base + row7]); \
        const int p = b_row0_pix + k4_idx; \
        const half4 in0 = read_imageh(src1, p + 0*k4_count); \
        const half4 in1 = read_imageh(src1, p + 1*k4_count); \
        const half4 in2 = read_imageh(src1, p + 2*k4_count); \
        const half4 in3 = read_imageh(src1, p + 3*k4_count); \
        const half4 in4 = read_imageh(src1, p + 4*k4_count); \
        const half4 in5 = read_imageh(src1, p + 5*k4_count); \
        const half4 in6 = read_imageh(src1, p + 6*k4_count); \
        const half4 in7 = read_imageh(src1, p + 7*k4_count); \
        half8 w; \
        w.s0 = ((bits8.s0 & 0x000F) - 8) * sc8.s0; \
        w.s1 = ((bits8.s1 & 0x000F) - 8) * sc8.s1; \
        w.s2 = ((bits8.s2 & 0x000F) - 8) * sc8.s2; \
        w.s3 = ((bits8.s3 & 0x000F) - 8) * sc8.s3; \
        w.s4 = ((bits8.s4 & 0x000F) - 8) * sc8.s4; \
        w.s5 = ((bits8.s5 & 0x000F) - 8) * sc8.s5; \
        w.s6 = ((bits8.s6 & 0x000F) - 8) * sc8.s6; \
        w.s7 = ((bits8.s7 & 0x000F) - 8) * sc8.s7; \
        acc0 += (half8)(in0.s0) * w; \
        acc1 += (half8)(in1.s0) * w; \
        acc2 += (half8)(in2.s0) * w; \
        acc3 += (half8)(in3.s0) * w; \
        acc4 += (half8)(in4.s0) * w; \
        acc5 += (half8)(in5.s0) * w; \
        acc6 += (half8)(in6.s0) * w; \
        acc7 += (half8)(in7.s0) * w; \
        w.s0 = (((bits8.s0 & 0x00F0) >> 4) - 8) * sc8.s0; \
        w.s1 = (((bits8.s1 & 0x00F0) >> 4) - 8) * sc8.s1; \
        w.s2 = (((bits8.s2 & 0x00F0) >> 4) - 8) * sc8.s2; \
        w.s3 = (((bits8.s3 & 0x00F0) >> 4) - 8) * sc8.s3; \
        w.s4 = (((bits8.s4 & 0x00F0) >> 4) - 8) * sc8.s4; \
        w.s5 = (((bits8.s5 & 0x00F0) >> 4) - 8) * sc8.s5; \
        w.s6 = (((bits8.s6 & 0x00F0) >> 4) - 8) * sc8.s6; \
        w.s7 = (((bits8.s7 & 0x00F0) >> 4) - 8) * sc8.s7; \
        acc0 += (half8)(in0.s1) * w; \
        acc1 += (half8)(in1.s1) * w; \
        acc2 += (half8)(in2.s1) * w; \
        acc3 += (half8)(in3.s1) * w; \
        acc4 += (half8)(in4.s1) * w; \
        acc5 += (half8)(in5.s1) * w; \
        acc6 += (half8)(in6.s1) * w; \
        acc7 += (half8)(in7.s1) * w; \
        w.s0 = (((bits8.s0 & 0x0F00) >> 8) - 8) * sc8.s0; \
        w.s1 = (((bits8.s1 & 0x0F00) >> 8) - 8) * sc8.s1; \
        w.s2 = (((bits8.s2 & 0x0F00) >> 8) - 8) * sc8.s2; \
        w.s3 = (((bits8.s3 & 0x0F00) >> 8) - 8) * sc8.s3; \
        w.s4 = (((bits8.s4 & 0x0F00) >> 8) - 8) * sc8.s4; \
        w.s5 = (((bits8.s5 & 0x0F00) >> 8) - 8) * sc8.s5; \
        w.s6 = (((bits8.s6 & 0x0F00) >> 8) - 8) * sc8.s6; \
        w.s7 = (((bits8.s7 & 0x0F00) >> 8) - 8) * sc8.s7; \
        acc0 += (half8)(in0.s2) * w; \
        acc1 += (half8)(in1.s2) * w; \
        acc2 += (half8)(in2.s2) * w; \
        acc3 += (half8)(in3.s2) * w; \
        acc4 += (half8)(in4.s2) * w; \
        acc5 += (half8)(in5.s2) * w; \
        acc6 += (half8)(in6.s2) * w; \
        acc7 += (half8)(in7.s2) * w; \
        w.s0 = (((bits8.s0 & 0xF000) >> 12) - 8) * sc8.s0; \
        w.s1 = (((bits8.s1 & 0xF000) >> 12) - 8) * sc8.s1; \
        w.s2 = (((bits8.s2 & 0xF000) >> 12) - 8) * sc8.s2; \
        w.s3 = (((bits8.s3 & 0xF000) >> 12) - 8) * sc8.s3; \
        w.s4 = (((bits8.s4 & 0xF000) >> 12) - 8) * sc8.s4; \
        w.s5 = (((bits8.s5 & 0xF000) >> 12) - 8) * sc8.s5; \
        w.s6 = (((bits8.s6 & 0xF000) >> 12) - 8) * sc8.s6; \
        w.s7 = (((bits8.s7 & 0xF000) >> 12) - 8) * sc8.s7; \
        acc0 += (half8)(in0.s3) * w; \
        acc1 += (half8)(in1.s3) * w; \
        acc2 += (half8)(in2.s3) * w; \
        acc3 += (half8)(in3.s3) * w; \
        acc4 += (half8)(in4.s3) * w; \
        acc5 += (half8)(in5.s3) * w; \
        acc6 += (half8)(in6.s3) * w; \
        acc7 += (half8)(in7.s3) * w; \
    } while (0)

    const int kb_count = hidden_dim >> 5;
    for (int kb = 0; kb < kb_count; ++kb) {
        const ulong d_base = (ulong) kb * (ulong) src_rows;
        const half8 sc = (half8)(
            d[d_base + row0], d[d_base + row1], d[d_base + row2], d[d_base + row3],
            d[d_base + row4], d[d_base + row5], d[d_base + row6], d[d_base + row7]);
        const int k4_base = kb << 3;
        for (int kk = 0; kk < 8; ++kk) {
            INDEXED_AB_BI8_NOSPLIT_ACCUM_K4(k4_base + kk, sc);
        }
    }

    #undef INDEXED_AB_BI8_NOSPLIT_ACCUM_K4

    if (out8 + 7 < out_rows && out_b_idx + 7 < n_rows) {
        __global float * outp = dst + out_off;
        vstore8(convert_float8(acc0), 0, outp + 0*out_rows);
        vstore8(convert_float8(acc1), 0, outp + 1*out_rows);
        vstore8(convert_float8(acc2), 0, outp + 2*out_rows);
        vstore8(convert_float8(acc3), 0, outp + 3*out_rows);
        vstore8(convert_float8(acc4), 0, outp + 4*out_rows);
        vstore8(convert_float8(acc5), 0, outp + 5*out_rows);
        vstore8(convert_float8(acc6), 0, outp + 6*out_rows);
        vstore8(convert_float8(acc7), 0, outp + 7*out_rows);
    }
}
)CLC";

#ifdef SPECINFER_QNN_OPENCL_EMBED_KERNELS
static std::string embedded_argsort_kernel_source() {
    return std::string {
        #include "argsort.cl.h"
    };
}

static std::string embedded_set_rows_kernel_source() {
    return std::string {
        #include "set_rows.cl.h"
    };
}

static std::string embedded_mul_mat_Ab_Bi_8x4_kernel_source() {
    return std::string {
        #include "mul_mat_Ab_Bi_8x4.cl.h"
    };
}
#endif

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

std::string cl_status_to_string(cl_int status) {
    switch (status) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: break;
    }

    std::ostringstream oss;
    oss << "OpenCL error " << status;
    return oss.str();
}

void throw_on_cl_error(cl_int status, const std::string & where) {
    if (status == CL_SUCCESS) {
        return;
    }

    std::ostringstream oss;
    oss << where << " failed: " << cl_status_to_string(status) << " (" << status << ")";
    fail(oss.str());
}

template <typename T>
void release_cl_handle(T & handle, cl_int (*release_fn)(T)) {
    if (handle != nullptr) {
        release_fn(handle);
        handle = nullptr;
    }
}

size_t round_up(size_t value, size_t multiple) {
    if (multiple == 0) {
        return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

int next_power_of_two(int value) {
    if (value <= 1) {
        return 1;
    }

    int padded = 1;
    while (padded < value) {
        padded <<= 1;
    }
    return padded;
}

uint16_t pattern_u16(uint32_t major, uint32_t minor, uint32_t seed0, uint32_t seed1) {
    uint32_t value = major * seed0;
    value ^= minor * seed1;
    value ^= (major + 17u) * (minor + 31u);
    return static_cast<uint16_t>(value & 0xffffu);
}

uint16_t float_to_half_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x007fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x00800000u;
        const uint32_t shifted = mant >> static_cast<uint32_t>(1 - exp);
        return static_cast<uint16_t>(sign | ((shifted + 0x00001000u) >> 13));
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x00001000u) >> 13));
}

std::string read_text_file(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        fail("failed to open file: " + path);
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

bool try_read_text_file(const std::string & path, std::string & out) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    out = oss.str();
    return true;
}

std::vector<int32_t> read_i32_list_file(const std::string & path) {
    const std::string text = read_text_file(path);
    std::vector<int32_t> values;

    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && text[i] != '-' && (text[i] < '0' || text[i] > '9')) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }

        int sign = 1;
        if (text[i] == '-') {
            sign = -1;
            ++i;
        }
        if (i >= text.size() || text[i] < '0' || text[i] > '9') {
            continue;
        }

        int64_t value = 0;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            value = value * 10 + static_cast<int64_t>(text[i] - '0');
            if (value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1) {
                fail("integer out of int32 range in ids file: " + path);
            }
            ++i;
        }
        value *= sign;
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
            fail("integer out of int32 range in ids file: " + path);
        }
        values.push_back(static_cast<int32_t>(value));
    }

    return values;
}

std::string format_bytes(size_t bytes) {
    static const char * suffixes[] = { "B", "KiB", "MiB", "GiB" };
    double value = static_cast<double>(bytes);
    size_t suffix = 0;
    while (value >= 1024.0 && suffix + 1 < (sizeof(suffixes) / sizeof(suffixes[0]))) {
        value /= 1024.0;
        ++suffix;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(value >= 100.0 ? 1 : 2) << value << ' ' << suffixes[suffix];
    return oss.str();
}

std::string format_lws(size_t lws) {
    if (lws == 0) {
        return "auto";
    }

    std::ostringstream oss;
    oss << lws;
    return oss.str();
}

std::string format_lws(size_t lx, size_t ly) {
    if (lx == 0 || ly == 0) {
        return "auto";
    }

    std::ostringstream oss;
    oss << lx << "x" << ly;
    return oss.str();
}

std::string format_gather_config(const GatherTuningConfig & config) {
    if (config.legacy_kernel) {
        std::ostringstream oss;
        oss << "legacy:" << format_lws(config.lx, config.ly);
        return oss.str();
    }

    std::ostringstream oss;
    oss << "wg=" << format_lws(config.lx, config.ly)
        << " rpt=" << config.rows_per_thread
        << " kpt=" << config.k4_per_thread
        << " ids=" << (config.use_local_ids ? "local" : "global");
    return oss.str();
}

std::string format_indexed_config(const IndexedMatvecTuningConfig & config) {
    std::ostringstream oss;
    if (config.abi_tile) {
        oss << "Ab_Bi_" << config.abi_n_tile << "x" << config.abi_m_tile
            << " local=1x" << config.wi_m << "x" << config.wi_k;
        if (config.abi_no_split) {
            oss << " nosplit";
        }
        if (config.abi_local_b) {
            oss << " localB";
        }
        if (config.abi_prefetch) {
            oss << " prefetch";
        }
    } else {
        oss << "lws=" << config.local_size
            << " rows/sg=" << config.rows_per_subgroup;
    }
    return oss.str();
}

std::vector<size_t> make_power_of_two_values(size_t limit) {
    std::vector<size_t> values;
    if (limit == 0) {
        return values;
    }

    const size_t capped_limit = std::min<size_t>(1024, limit);
    for (size_t value = 1; value <= capped_limit; value <<= 1) {
        values.push_back(value);
        if (value > (std::numeric_limits<size_t>::max() >> 1)) {
            break;
        }
    }

    return values;
}

std::vector<size_t> make_linear_values(size_t limit) {
    std::vector<size_t> values;
    values.reserve(limit);
    for (size_t value = 1; value <= limit; ++value) {
        values.push_back(value);
    }
    return values;
}

void print_usage(const char * argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --platform N          OpenCL platform index (default: 0)\n"
        << "  --device N            OpenCL device index within the platform (default: 0)\n"
        << "  --n-scores N          Selector score count for top-k search (default: 128256)\n"
        << "  --src-rows N          Source Q4_0 row count for gather (default: n-scores)\n"
        << "  --top-k N             Selector shortlist size before gather (default: 64)\n"
        << "  --hidden-dim N        Reduced LM-head hidden dimension (default: 4096)\n"
        << "  --lmhead-batch N      Reduced LM-head batch rows / N dimension (default: 1)\n"
        << "  --gather-rows N       Gather row count after padding (default: max(top-k, 512), rounded to 8)\n"
        << "  --min-gather-rows N   Minimum padded gather rows if --gather-rows is omitted (default: 512)\n"
        << "  --warmup N            Warmup iterations per candidate (default: 1)\n"
        << "  --iters N             Timed iterations per candidate (default: 5)\n"
        << "                        Search uses power-of-two local work sizes up to 1024;\n"
        << "                        'auto' is measured as a baseline for top-k/id-sort.\n"
        << "                        Gather also sweeps rows/thread={1,2,4,8},\n"
        << "                        k4/thread={1,2,4}, and ids staging={global,local}.\n"
        << "  --seed N              RNG seed for synthetic score generation (default: 42)\n"
        << "  --kernel-dir PATH     Optional directory containing argsort.cl and set_rows.cl\n"
        << "  --ids-file PATH       Optional JSON/text int list to use as gather/indexed row ids\n"
        << "  --search MODE         all | topk | bucket-topk | id-sort | gather | indexed (default: all)\n"
        << "  --allow-non-power-local\n"
        << "                        Also try exploratory non-power-of-two indexed Ab_Bi local sizes\n"
        << "  --verbose             Print every candidate result\n"
        << "  --help                Show this message\n";
}

int parse_int_arg(const char * argv0, const char * name, int argc, char ** argv, int & index) {
    if (index + 1 >= argc) {
        std::ostringstream oss;
        oss << argv0 << ": missing value for " << name;
        fail(oss.str());
    }

    char * end = nullptr;
    const long value = std::strtol(argv[index + 1], &end, 10);
    if (end == nullptr || *end != '\0') {
        std::ostringstream oss;
        oss << argv0 << ": invalid integer for " << name << ": " << argv[index + 1];
        fail(oss.str());
    }

    ++index;
    return static_cast<int>(value);
}

std::string parse_string_arg(const char * argv0, const char * name, int argc, char ** argv, int & index) {
    if (index + 1 >= argc) {
        std::ostringstream oss;
        oss << argv0 << ": missing value for " << name;
        fail(oss.str());
    }

    ++index;
    return argv[index];
}

Options parse_options(int argc, char ** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--platform") {
            options.platform_index = parse_int_arg(argv[0], "--platform", argc, argv, i);
        } else if (arg == "--device") {
            options.device_index = parse_int_arg(argv[0], "--device", argc, argv, i);
        } else if (arg == "--n-scores") {
            options.n_scores = parse_int_arg(argv[0], "--n-scores", argc, argv, i);
        } else if (arg == "--src-rows") {
            options.src_rows = parse_int_arg(argv[0], "--src-rows", argc, argv, i);
        } else if (arg == "--top-k") {
            options.top_k = parse_int_arg(argv[0], "--top-k", argc, argv, i);
        } else if (arg == "--hidden-dim") {
            options.hidden_dim = parse_int_arg(argv[0], "--hidden-dim", argc, argv, i);
        } else if (arg == "--lmhead-batch") {
            options.lmhead_batch = parse_int_arg(argv[0], "--lmhead-batch", argc, argv, i);
        } else if (arg == "--gather-rows") {
            options.gather_rows = parse_int_arg(argv[0], "--gather-rows", argc, argv, i);
        } else if (arg == "--min-gather-rows") {
            options.min_gather_rows = parse_int_arg(argv[0], "--min-gather-rows", argc, argv, i);
        } else if (arg == "--warmup") {
            options.warmup = parse_int_arg(argv[0], "--warmup", argc, argv, i);
        } else if (arg == "--iters") {
            options.iters = parse_int_arg(argv[0], "--iters", argc, argv, i);
        } else if (arg == "--seed") {
            options.seed = parse_int_arg(argv[0], "--seed", argc, argv, i);
        } else if (arg == "--kernel-dir") {
            options.kernel_dir = parse_string_arg(argv[0], "--kernel-dir", argc, argv, i);
        } else if (arg == "--ids-file") {
            options.ids_file = parse_string_arg(argv[0], "--ids-file", argc, argv, i);
        } else if (arg == "--allow-non-power-local") {
            options.allow_non_power_of_two_local = true;
        } else if (arg == "--search") {
            if (i + 1 >= argc) {
                fail("missing value for --search");
            }
            const std::string mode = argv[++i];
            options.tune_topk = false;
            options.tune_bucket_topk = false;
            options.tune_id_sort = false;
            options.tune_gather = false;
            options.tune_indexed = false;

            if (mode == "all") {
                options.tune_topk = true;
                options.tune_bucket_topk = true;
                options.tune_id_sort = true;
                options.tune_gather = true;
                options.tune_indexed = true;
            } else if (mode == "topk") {
                options.tune_topk = true;
            } else if (mode == "bucket-topk") {
                options.tune_bucket_topk = true;
            } else if (mode == "id-sort") {
                options.tune_id_sort = true;
            } else if (mode == "gather") {
                options.tune_gather = true;
            } else if (mode == "indexed") {
                options.tune_indexed = true;
            } else {
                fail("invalid value for --search: " + mode);
            }
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            options.show_help = true;
        } else {
            fail("unknown argument: " + arg);
        }
    }

    if (options.src_rows < 0) {
        options.src_rows = options.n_scores;
    }

    if (options.gather_rows < 0) {
        options.gather_rows = std::max(options.top_k, options.min_gather_rows);
    }

    options.gather_rows = static_cast<int>(round_up(static_cast<size_t>(options.gather_rows), 8));

    if (options.n_scores <= 0) {
        fail("--n-scores must be > 0");
    }
    if (options.src_rows <= 0) {
        fail("--src-rows must be > 0");
    }
    if (options.top_k <= 0) {
        fail("--top-k must be > 0");
    }
    if (options.src_rows < options.n_scores) {
        fail("--src-rows must be >= --n-scores to keep top-k ids in range");
    }
    if (options.hidden_dim <= 0 || (options.hidden_dim % 32) != 0) {
        fail("--hidden-dim must be > 0 and divisible by 32");
    }
    if (options.lmhead_batch <= 0 || options.lmhead_batch > 8) {
        fail("--lmhead-batch must be in [1, 8] for the current 8-row tile benchmark");
    }
    if (options.gather_rows < options.top_k) {
        fail("--gather-rows must be >= --top-k");
    }
    if (options.warmup < 0) {
        fail("--warmup must be >= 0");
    }
    if (options.iters <= 0) {
        fail("--iters must be > 0");
    }
    if (options.platform_index < 0 || options.device_index < 0) {
        fail("--platform and --device must be >= 0");
    }

    return options;
}

std::vector<int32_t> cpu_topk_desc_indices(const std::vector<float> & scores, int top_k) {
    std::vector<int32_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    const int output_count = std::min<int>(top_k, static_cast<int>(indices.size()));
    const auto cmp = [&scores](int32_t lhs, int32_t rhs) {
        if (scores[lhs] != scores[rhs]) {
            return scores[lhs] > scores[rhs];
        }
        return lhs < rhs;
    };

    std::partial_sort(indices.begin(), indices.begin() + output_count, indices.end(), cmp);
    indices.resize(output_count);
    return indices;
}

template <typename Fn>
std::pair<double, double> benchmark_iterations(int warmup, int iters, Fn && fn) {
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::infinity();

    for (int i = 0; i < warmup + iters; ++i) {
        const double ms = fn();
        if (i >= warmup) {
            total_ms += ms;
            min_ms = std::min(min_ms, ms);
        }
    }

    return { total_ms / static_cast<double>(iters), min_ms };
}

class OpenCLTuner {
public:
    explicit OpenCLTuner(const Options & options)
        : options_(options),
          padded_score_count_(next_power_of_two(options_.n_scores)),
          output_count_(std::min(options_.top_k, options_.n_scores)),
          id_sort_padded_count_(next_power_of_two(output_count_)),
          gather_rows_(options_.gather_rows),
          src_rows_(options_.src_rows),
          k4_count_(options_.hidden_dim / 4),
          kb_count_(options_.hidden_dim / 32) {
        initialize_host_data();
        initialize_opencl();
        build_kernels();
        initialize_buffers();
    }

    ~OpenCLTuner() {
        release_cl_handle(indexed_out_buffer_, clReleaseMemObject);
        release_cl_handle(reference_out_buffer_, clReleaseMemObject);
        release_cl_handle(dense_out_buffer_, clReleaseMemObject);
        release_cl_handle(dense_b_image_, clReleaseMemObject);
        release_cl_handle(dense_b_half_buffer_, clReleaseMemObject);
        release_cl_handle(hidden_buffer_, clReleaseMemObject);
        release_cl_handle(dst_d_buffer_, clReleaseMemObject);
        release_cl_handle(dst_q_buffer_, clReleaseMemObject);
        release_cl_handle(gather_ids_buffer_, clReleaseMemObject);
        release_cl_handle(src_d_buffer_, clReleaseMemObject);
        release_cl_handle(src_q_buffer_, clReleaseMemObject);
        release_cl_handle(id_sort_scratch_buffer_, clReleaseMemObject);
        release_cl_handle(working_ids_buffer_, clReleaseMemObject);
        release_cl_handle(baseline_ids_buffer_, clReleaseMemObject);
        release_cl_handle(working_indices_buffer_, clReleaseMemObject);
        release_cl_handle(working_scores_buffer_, clReleaseMemObject);
        release_cl_handle(baseline_scores_buffer_, clReleaseMemObject);

        release_cl_handle(init_pattern_kernel_, clReleaseKernel);
        release_cl_handle(indexed_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi4_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi_nosplit_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi4_nosplit_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi4_n2_nosplit_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi_lbtile_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi_prefetch_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi_n8m1_nosplit_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi_n8m2_nosplit_matvec_kernel_, clReleaseKernel);
        release_cl_handle(indexed_abi_n8m8_nosplit_matvec_kernel_, clReleaseKernel);
        release_cl_handle(dense_matvec_kernel_, clReleaseKernel);
        release_cl_handle(production_dense_matvec_kernel_, clReleaseKernel);
        release_cl_handle(gather_kernel_, clReleaseKernel);
        release_cl_handle(sort_i32_kernel_, clReleaseKernel);
        release_cl_handle(sort_f32_i32_kernel_, clReleaseKernel);
        release_cl_handle(fill_i32_kernel_, clReleaseKernel);
        release_cl_handle(init_i32_range_kernel_, clReleaseKernel);
        release_cl_handle(gather_program_, clReleaseProgram);
        release_cl_handle(matvec_program_, clReleaseProgram);
        release_cl_handle(production_dense_matvec_program_, clReleaseProgram);
        release_cl_handle(argsort_program_, clReleaseProgram);

        release_cl_handle(queue_, clReleaseCommandQueue);
        release_cl_handle(context_, clReleaseContext);
    }

    void print_problem_summary() const {
        const size_t score_bytes = sizeof(float) * static_cast<size_t>(padded_score_count_);
        const size_t src_q_bytes = sizeof(uint16_t) * static_cast<size_t>(k4_count_) * static_cast<size_t>(src_rows_);
        const size_t src_d_bytes = sizeof(uint16_t) * static_cast<size_t>(kb_count_) * static_cast<size_t>(src_rows_);
        const size_t dst_q_bytes = sizeof(uint16_t) * static_cast<size_t>(k4_count_) * static_cast<size_t>(gather_rows_);
        const size_t dst_d_bytes = sizeof(uint16_t) * static_cast<size_t>(kb_count_) * static_cast<size_t>(gather_rows_);

        std::cout << "OpenCL platform : " << platform_name_ << "\n";
        std::cout << "OpenCL device   : " << device_name_ << "\n";
        std::cout << "Device memory   : " << format_bytes(static_cast<size_t>(global_mem_bytes_)) << "\n";
        std::cout << "Local memory    : " << format_bytes(static_cast<size_t>(device_local_mem_bytes_)) << "\n";
        std::cout << "Max work-group  : " << device_max_work_group_size_ << "\n";
        std::cout << "Work-item sizes : ";
        for (size_t i = 0; i < device_max_work_item_sizes_.size(); ++i) {
            if (i != 0) {
                std::cout << " x ";
            }
            std::cout << device_max_work_item_sizes_[i];
        }
        std::cout << "\n";
        std::cout << "Problem         : scores=" << options_.n_scores
                  << " padded_scores=" << padded_score_count_
                  << " top_k=" << output_count_
                  << " src_rows=" << src_rows_
                  << " gather_rows=" << gather_rows_
                  << " hidden_dim=" << options_.hidden_dim
                  << " lmhead_batch=" << options_.lmhead_batch
                  << " k4_count=" << k4_count_
                  << " kb_count=" << kb_count_;
        if (!options_.ids_file.empty()) {
            std::cout << " ids_file=" << options_.ids_file;
        }
        std::cout << "\n";
        std::cout << "Buffer sizes    : scores=" << format_bytes(score_bytes)
                  << " src_q=" << format_bytes(src_q_bytes)
                  << " src_d=" << format_bytes(src_d_bytes)
                  << " dst_q=" << format_bytes(dst_q_bytes)
                  << " dst_d=" << format_bytes(dst_d_bytes) << "\n";
        std::cout << "Kernel limits   : topk=" << sort_f32_i32_kernel_max_wgs_
                  << " id-sort=" << sort_i32_kernel_max_wgs_
                  << " gather-legacy=" << gather_kernel_max_wgs_
                  << " dense-matvec=" << production_dense_matvec_kernel_max_wgs_
                  << " indexed-matvec=" << indexed_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi=" << indexed_abi_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi4=" << indexed_abi4_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi-nosplit=" << indexed_abi_nosplit_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi4-nosplit=" << indexed_abi4_nosplit_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi4-n2-nosplit=" << indexed_abi4_n2_nosplit_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi-localB=" << indexed_abi_lbtile_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi-prefetch=" << indexed_abi_prefetch_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi-n8m1-nosplit=" << indexed_abi_n8m1_nosplit_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi-n8m2-nosplit=" << indexed_abi_n8m2_nosplit_matvec_kernel_max_wgs_
                  << " indexed-Ab_Bi-n8m8-nosplit=" << indexed_abi_n8m8_nosplit_matvec_kernel_max_wgs_
                  << " preferred-multiple=" << gather_kernel_preferred_multiple_ << "\n";
        std::cout << "Kernel source   : " << kernel_source_description_ << ", gather-tuner=embedded\n";
    }

    OneDimResult search_topk() {
        const auto candidates = make_1d_candidates(sort_f32_i32_kernel_max_wgs_);
        if (candidates.empty()) {
            fail("no valid top-k local size candidates");
        }

        std::cout << "\n[topk] Searching power-of-two local work sizes up to 1024 for selector bitonic sort\n";

        std::vector<OneDimResult> results;
        for (size_t lws : candidates) {
            try {
                const auto [avg_ms, min_ms] = benchmark_iterations(options_.warmup, options_.iters, [&]() {
                    return run_topk_once(lws);
                });
                validate_topk_output();

                results.push_back({ lws, avg_ms, min_ms });
                if (options_.verbose) {
                    std::cout << "  lws=" << std::setw(5) << format_lws(lws)
                              << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms
                              << " min_ms=" << min_ms << "\n";
                }
            } catch (const std::exception & e) {
                if (options_.verbose) {
                    std::cout << "  lws=" << std::setw(5) << format_lws(lws)
                              << " invalid (" << e.what() << ")\n";
                }
            }
        }

        if (results.empty()) {
            fail("no valid top-k candidates finished successfully");
        }

        std::sort(results.begin(), results.end(), [](const OneDimResult & lhs, const OneDimResult & rhs) {
            return lhs.avg_ms < rhs.avg_ms;
        });

        print_top_1d_results(results, "topk", "auto");
        return results.front();
    }

    OneDimResult search_bucket_topk() {
        const size_t limit = std::min(topk_hist_kernel_max_wgs_, topk_compact_kernel_max_wgs_);
        const auto candidates = make_1d_candidates(limit);
        if (candidates.empty()) {
            fail("no valid bucket-topk local size candidates");
        }

        std::cout << "\n[bucket-topk] Searching power-of-two local work sizes for histogram/compact kernels\n";

        std::vector<OneDimResult> results;
        for (size_t lws : candidates) {
            try {
                const auto [avg_ms, min_ms] = benchmark_iterations(options_.warmup, options_.iters, [&]() {
                    return run_bucket_topk_once(lws);
                });
                validate_bucket_topk_output();

                results.push_back({ lws, avg_ms, min_ms });
                if (options_.verbose) {
                    std::cout << "  lws=" << std::setw(5) << format_lws(lws)
                              << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms
                              << " min_ms=" << min_ms << "\n";
                }
            } catch (const std::exception & e) {
                if (options_.verbose) {
                    std::cout << "  lws=" << std::setw(5) << format_lws(lws)
                              << " invalid (" << e.what() << ")\n";
                }
            }
        }

        if (results.empty()) {
            fail("no valid bucket-topk candidates finished successfully");
        }

        std::sort(results.begin(), results.end(), [](const OneDimResult & lhs, const OneDimResult & rhs) {
            return lhs.avg_ms < rhs.avg_ms;
        });

        print_top_1d_results(results, "bucket-topk", "auto");
        return results.front();
    }

    OneDimResult search_id_sort() {
        const auto candidates = make_1d_candidates(sort_i32_kernel_max_wgs_);
        if (candidates.empty()) {
            fail("no valid id-sort local size candidates");
        }

        std::cout << "\n[id-sort] Searching power-of-two local work sizes up to 1024 for ascending gather-id reorder\n";

        std::vector<OneDimResult> results;
        for (size_t lws : candidates) {
            try {
                const auto [avg_ms, min_ms] = benchmark_iterations(options_.warmup, options_.iters, [&]() {
                    return run_id_sort_once(lws);
                });
                validate_id_sort_output();

                results.push_back({ lws, avg_ms, min_ms });
                if (options_.verbose) {
                    std::cout << "  lws=" << std::setw(5) << format_lws(lws)
                              << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms
                              << " min_ms=" << min_ms << "\n";
                }
            } catch (const std::exception & e) {
                if (options_.verbose) {
                    std::cout << "  lws=" << std::setw(5) << format_lws(lws)
                              << " invalid (" << e.what() << ")\n";
                }
            }
        }

        if (results.empty()) {
            fail("no valid id-sort candidates finished successfully");
        }

        std::sort(results.begin(), results.end(), [](const OneDimResult & lhs, const OneDimResult & rhs) {
            return lhs.avg_ms < rhs.avg_ms;
        });

        print_top_1d_results(results, "id-sort", "auto");
        return results.front();
    }

    GatherResult search_gather() {
        std::cout << "\n[gather] Searching legacy baseline plus power-of-two WG/tile configs for Q4_0 gather\n";

        const auto configs = make_gather_tuning_configs();
        if (configs.empty()) {
            fail("no valid tunable gather configurations fit the runtime limits");
        }

        std::cout << "  Structural candidates: " << configs.size() << " + 1 legacy baseline\n";

        std::vector<GatherResult> results;
        const GatherTuningConfig legacy_config = {
            64,
            2,
            1,
            1,
            false,
            true,
        };

        try {
            const auto [avg_ms, min_ms] = benchmark_iterations(options_.warmup, options_.iters, [&]() {
                return run_legacy_gather_once(legacy_config.lx, legacy_config.ly);
            });
            validate_gather_output();

            results.push_back({ legacy_config, avg_ms, min_ms });
            if (options_.verbose) {
                std::cout << "  config=" << format_gather_config(legacy_config)
                          << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms
                          << " min_ms=" << min_ms << "\n";
            }
        } catch (const std::exception & e) {
            fail(std::string("legacy gather baseline failed: ") + e.what());
        }

        for (const auto & config : configs) {
            try {
                results.push_back(benchmark_tunable_gather_config(config));
                if (options_.verbose) {
                    const GatherResult & result = results.back();
                    std::cout << "  config=" << format_gather_config(result.config)
                              << " avg_ms=" << std::fixed << std::setprecision(3) << result.avg_ms
                              << " min_ms=" << result.min_ms << "\n";
                }
            } catch (const std::exception & e) {
                if (options_.verbose) {
                    std::cout << "  config=" << format_gather_config(config)
                              << " invalid (" << e.what() << ")\n";
                }
            }
        }

        if (results.empty()) {
            fail("no valid gather candidates finished successfully");
        }

        std::sort(results.begin(), results.end(), [](const GatherResult & lhs, const GatherResult & rhs) {
            return lhs.avg_ms < rhs.avg_ms;
        });

        print_top_gather_results(results);
        return results.front();
    }

    std::pair<MatvecResult, MatvecResult> search_indexed_matvec() {
        std::cout << "\n[indexed] Comparing packed dense Q4_0 matvec with tuned direct indexed Q4_0 matvec\n";

        // Build the packed dense matrix once so dense and indexed use identical
        // row ids and only differ in whether the rows are pre-gathered.
        (void) run_legacy_gather_once(64, 2);

        const auto [dense_avg_ms, dense_min_ms] = benchmark_iterations(options_.warmup, options_.iters, [&]() {
            return run_dense_matvec_once();
        });
        validate_dense_matvec_finite();
        if (options_.lmhead_batch == 1) {
            (void) run_reference_dense_matvec_once();
        }

        const auto configs = make_indexed_matvec_tuning_configs();
        const auto abi_configs = make_indexed_abi_matvec_tuning_configs();
        if (configs.empty() && abi_configs.empty()) {
            fail("no valid indexed matvec configurations fit the runtime limits");
        }

        std::cout << "  Indexed structural candidates: " << configs.size()
                  << " subgroup + " << abi_configs.size() << " Ab_Bi-tile\n";

        std::vector<MatvecResult> indexed_results;
        for (const auto & config : configs) {
            try {
                indexed_results.push_back(benchmark_indexed_matvec_config(config));
                if (options_.verbose) {
                    const auto & result = indexed_results.back();
                    std::cout << "  config=" << format_indexed_config(config)
                              << " avg_ms=" << std::fixed << std::setprecision(3) << result.avg_ms
                              << " min_ms=" << result.min_ms << "\n";
                }
            } catch (const std::exception & e) {
                if (options_.verbose) {
                    std::cout << "  config=" << format_indexed_config(config)
                              << " invalid (" << e.what() << ")\n";
                }
            }
        }
        for (const auto & config : abi_configs) {
            try {
                indexed_results.push_back(benchmark_indexed_abi_matvec_config(config));
                if (options_.verbose) {
                    const auto & result = indexed_results.back();
                    std::cout << "  config=" << format_indexed_config(config)
                              << " avg_ms=" << std::fixed << std::setprecision(3) << result.avg_ms
                              << " min_ms=" << result.min_ms << "\n";
                }
            } catch (const std::exception & e) {
                if (options_.verbose) {
                    std::cout << "  config=" << format_indexed_config(config)
                              << " invalid (" << e.what() << ")\n";
                }
            }
        }

        if (indexed_results.empty()) {
            fail("no valid indexed matvec candidates finished successfully");
        }

        std::sort(indexed_results.begin(), indexed_results.end(), [](const MatvecResult & lhs, const MatvecResult & rhs) {
            return lhs.avg_ms < rhs.avg_ms;
        });

        MatvecResult dense_result{ "dense-packed-Ab_Bi_8x4", {}, dense_avg_ms, dense_min_ms };
        MatvecResult indexed_result = indexed_results.front();

        std::cout << "  Dense packed rows : " << gather_rows_
                  << " avg_ms=" << std::fixed << std::setprecision(3) << dense_result.avg_ms
                  << " min_ms=" << dense_result.min_ms << "\n";
        std::cout << "  Best indexed rows : " << gather_rows_
                  << " from src_rows=" << src_rows_
                  << " " << format_indexed_config(indexed_result.indexed_config)
                  << " avg_ms=" << std::fixed << std::setprecision(3) << indexed_result.avg_ms
                  << " min_ms=" << indexed_result.min_ms << "\n";
        std::cout << "  Best indexed configs:\n";
        const size_t topn = std::min<size_t>(5, indexed_results.size());
        for (size_t i = 0; i < topn; ++i) {
            std::cout << "    " << std::setw(2) << (i + 1)
                      << ". " << format_indexed_config(indexed_results[i].indexed_config)
                      << " avg_ms=" << std::fixed << std::setprecision(3) << indexed_results[i].avg_ms
                      << " min_ms=" << indexed_results[i].min_ms << "\n";
        }
        auto nosplit_it = std::find_if(indexed_results.begin(), indexed_results.end(),
                [](const MatvecResult & result) {
                    return result.indexed_config.abi_no_split;
                });
        if (nosplit_it != indexed_results.end()) {
            std::cout << "  Best nosplit      : " << format_indexed_config(nosplit_it->indexed_config)
                      << " avg_ms=" << std::fixed << std::setprecision(3) << nosplit_it->avg_ms
                      << " min_ms=" << nosplit_it->min_ms << "\n";
        }
        std::cout << "  Indexed slowdown  : " << std::fixed << std::setprecision(3)
                  << (indexed_result.avg_ms / dense_result.avg_ms) << "x vs dense packed\n";

        return { dense_result, indexed_result };
    }

private:
    void initialize_host_data() {
        std::mt19937 rng(static_cast<uint32_t>(options_.seed));
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

        padded_scores_.assign(static_cast<size_t>(padded_score_count_), -std::numeric_limits<float>::infinity());
        for (int i = 0; i < options_.n_scores; ++i) {
            float value = dist(rng);
            if ((i % 257) == 0) {
                value = 1.0f + static_cast<float>(i % 17) * 0.01f;
            }
            padded_scores_[static_cast<size_t>(i)] = value;
        }

        topk_desc_ref_ = cpu_topk_desc_indices(padded_scores_, output_count_);
        topk_asc_ref_ = topk_desc_ref_;
        std::sort(topk_asc_ref_.begin(), topk_asc_ref_.end());

        std::vector<int32_t> source_ids = topk_asc_ref_;
        if (!options_.ids_file.empty()) {
            source_ids = read_i32_list_file(options_.ids_file);
            if (static_cast<int>(source_ids.size()) < output_count_) {
                std::ostringstream oss;
                oss << "--ids-file contains " << source_ids.size()
                    << " ids, but --top-k requires " << output_count_;
                fail(oss.str());
            }
            for (int i = 0; i < output_count_; ++i) {
                const int32_t id = source_ids[static_cast<size_t>(i)];
                if (id < 0 || id >= src_rows_) {
                    std::ostringstream oss;
                    oss << "--ids-file id out of range at position " << i
                        << ": " << id << " (src_rows=" << src_rows_ << ")";
                    fail(oss.str());
                }
            }
        }

        gather_ids_.resize(static_cast<size_t>(gather_rows_));
        for (int i = 0; i < output_count_; ++i) {
            gather_ids_[static_cast<size_t>(i)] = source_ids[static_cast<size_t>(i)];
        }
        for (int i = output_count_; i < gather_rows_; ++i) {
            gather_ids_[static_cast<size_t>(i)] = source_ids[static_cast<size_t>(output_count_ - 1)];
        }

        hidden_.resize(static_cast<size_t>(options_.hidden_dim) * static_cast<size_t>(options_.lmhead_batch));
        for (float & value : hidden_) {
            value = dist(rng) * 0.125f;
        }
    }

    void initialize_opencl() {
        cl_uint platform_count = 0;
        throw_on_cl_error(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
        if (platform_count == 0) {
            fail("no OpenCL platforms found");
        }

        std::vector<cl_platform_id> platforms(platform_count);
        throw_on_cl_error(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");

        if (options_.platform_index >= static_cast<int>(platforms.size())) {
            fail("platform index out of range");
        }
        platform_ = platforms[static_cast<size_t>(options_.platform_index)];
        platform_name_ = query_platform_string(platform_, CL_PLATFORM_NAME);

        cl_uint device_count = 0;
        throw_on_cl_error(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count), "clGetDeviceIDs(count)");
        if (device_count == 0) {
            fail("no OpenCL devices found on selected platform");
        }

        std::vector<cl_device_id> devices(device_count);
        throw_on_cl_error(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr), "clGetDeviceIDs(list)");

        if (options_.device_index >= static_cast<int>(devices.size())) {
            fail("device index out of range");
        }
        device_ = devices[static_cast<size_t>(options_.device_index)];
        device_name_ = query_device_string(device_, CL_DEVICE_NAME);

        cl_int status = CL_SUCCESS;
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &status);
        throw_on_cl_error(status, "clCreateContext");

        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &status);
        throw_on_cl_error(status, "clCreateCommandQueue");

        throw_on_cl_error(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                          sizeof(device_max_work_group_size_),
                                          &device_max_work_group_size_, nullptr),
                          "clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)");

        cl_uint max_dims = 0;
        throw_on_cl_error(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                          sizeof(max_dims), &max_dims, nullptr),
                          "clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)");
        device_max_work_item_sizes_.resize(max_dims);
        throw_on_cl_error(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                          sizeof(size_t) * static_cast<size_t>(max_dims),
                                          device_max_work_item_sizes_.data(), nullptr),
                          "clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES)");

        throw_on_cl_error(clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,
                                          sizeof(global_mem_bytes_), &global_mem_bytes_, nullptr),
                          "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE)");
        throw_on_cl_error(clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE,
                                          sizeof(device_local_mem_bytes_), &device_local_mem_bytes_, nullptr),
                          "clGetDeviceInfo(CL_DEVICE_LOCAL_MEM_SIZE)");
    }

    void build_kernels() {
        std::string set_rows_src;
        std::string production_dense_q4_src;

        const std::string kernel_dir = options_.kernel_dir.empty()
                ? std::string(SPECINFER_QNN_OPENCL_KERNEL_DIR)
                : options_.kernel_dir;
        const bool loaded_set_rows_from_files =
                try_read_text_file(kernel_dir + "/set_rows.cl", set_rows_src);
        const bool loaded_dense_q4_from_files =
                try_read_text_file(kernel_dir + "/mul_mat_Ab_Bi_8x4.cl", production_dense_q4_src);

        if (loaded_set_rows_from_files) {
            kernel_source_description_ = "argsort=minimal-embedded, set_rows=filesystem (" + kernel_dir + ")";
        } else {
#ifdef SPECINFER_QNN_OPENCL_EMBED_KERNELS
            set_rows_src = embedded_set_rows_kernel_source();
            kernel_source_description_ = "argsort=minimal-embedded, set_rows=embedded";
#else
            fail("failed to open OpenCL kernel file set_rows.cl from: " + kernel_dir);
#endif
        }
        if (!loaded_dense_q4_from_files) {
#ifdef SPECINFER_QNN_OPENCL_EMBED_KERNELS
            production_dense_q4_src = embedded_mul_mat_Ab_Bi_8x4_kernel_source();
#else
            fail("failed to open OpenCL kernel file mul_mat_Ab_Bi_8x4.cl from: " + kernel_dir);
#endif
        }
        kernel_source_description_ += loaded_dense_q4_from_files
                ? ", dense=mul_mat_Ab_Bi_8x4(filesystem)"
                : ", dense=mul_mat_Ab_Bi_8x4(embedded)";

        argsort_program_ = build_program({ std::string(kMinimalArgsortKernels) }, "argsort");
        gather_program_ = build_program({ std::string(kPatternInitKernels), set_rows_src }, "set_rows");
        matvec_program_ = build_program(
                { std::string(kQ4MatvecKernels) },
                "q4-matvec",
                "-cl-std=CL2.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math");
        production_dense_matvec_program_ = build_program(
                { production_dense_q4_src },
                "production-dense-q4-matvec",
                "-cl-std=CL2.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math");

        init_i32_range_kernel_ = create_kernel(argsort_program_, "kernel_init_i32_range");
        fill_i32_kernel_ = create_kernel(argsort_program_, "kernel_fill_i32");
        sort_f32_i32_kernel_ = create_kernel(argsort_program_, "kernel_bitonic_sort_step_f32_i32");
        sort_i32_kernel_ = create_kernel(argsort_program_, "kernel_bitonic_sort_step_i32");
        topk_hist_kernel_    = create_kernel(argsort_program_, "kernel_topk_hist_f32");
        topk_compact_kernel_ = create_kernel(argsort_program_, "kernel_topk_compact_f32");
        gather_kernel_ = create_kernel(gather_program_, "kernel_gather_rows_q4_0_transposed_i32");
        init_pattern_kernel_ = create_kernel(gather_program_, "kernel_init_pattern_u16");
        dense_matvec_kernel_ = create_kernel(matvec_program_, "kernel_dense_q4_0_matvec_8x");
        indexed_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_8x");
        indexed_abi_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_8x4");
        indexed_abi4_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_4x4");
        indexed_abi_nosplit_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_8x4_nosplit");
        indexed_abi4_nosplit_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_4x4_nosplit");
        indexed_abi4_n2_nosplit_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_4x2_nosplit");
        indexed_abi_lbtile_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_8x4_lbtile");
        indexed_abi_prefetch_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_8x4_nosplit_prefetch");
        indexed_abi_n8m1_nosplit_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_8x1_nosplit");
        indexed_abi_n8m2_nosplit_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_8x2_nosplit");
        indexed_abi_n8m8_nosplit_matvec_kernel_ = create_kernel(matvec_program_, "kernel_indexed_q4_0_matvec_Ab_Bi_8x8_nosplit");
        production_dense_matvec_kernel_ = create_kernel(production_dense_matvec_program_, "kernel_mul_mat_Ab_Bi_8x4");

        sort_f32_i32_kernel_max_wgs_ = query_kernel_wgs(sort_f32_i32_kernel_);
        sort_i32_kernel_max_wgs_ = query_kernel_wgs(sort_i32_kernel_);
        topk_hist_kernel_max_wgs_    = query_kernel_wgs(topk_hist_kernel_);
        topk_compact_kernel_max_wgs_ = query_kernel_wgs(topk_compact_kernel_);
        gather_kernel_max_wgs_ = query_kernel_wgs(gather_kernel_);
        gather_kernel_preferred_multiple_ = query_kernel_preferred_multiple(gather_kernel_);
        dense_matvec_kernel_max_wgs_ = query_kernel_wgs(dense_matvec_kernel_);
        indexed_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_matvec_kernel_);
        indexed_abi_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi_matvec_kernel_);
        indexed_abi4_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi4_matvec_kernel_);
        indexed_abi_nosplit_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi_nosplit_matvec_kernel_);
        indexed_abi4_nosplit_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi4_nosplit_matvec_kernel_);
        indexed_abi4_n2_nosplit_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi4_n2_nosplit_matvec_kernel_);
        indexed_abi_lbtile_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi_lbtile_matvec_kernel_);
        indexed_abi_prefetch_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi_prefetch_matvec_kernel_);
        indexed_abi_n8m1_nosplit_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi_n8m1_nosplit_matvec_kernel_);
        indexed_abi_n8m2_nosplit_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi_n8m2_nosplit_matvec_kernel_);
        indexed_abi_n8m8_nosplit_matvec_kernel_max_wgs_ = query_kernel_wgs(indexed_abi_n8m8_nosplit_matvec_kernel_);
        production_dense_matvec_kernel_max_wgs_ = query_kernel_wgs(production_dense_matvec_kernel_);
    }

    void initialize_buffers() {
        baseline_scores_buffer_ = create_buffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * static_cast<size_t>(padded_score_count_),
                padded_scores_.data(),
                "baseline_scores_buffer");
        working_scores_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(float) * static_cast<size_t>(padded_score_count_),
                nullptr,
                "working_scores_buffer");
        working_indices_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(int32_t) * static_cast<size_t>(padded_score_count_),
                nullptr,
                "working_indices_buffer");

        baseline_ids_buffer_ = create_buffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(int32_t) * static_cast<size_t>(output_count_),
                topk_desc_ref_.data(),
                "baseline_ids_buffer");
        working_ids_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(int32_t) * static_cast<size_t>(output_count_),
                nullptr,
                "working_ids_buffer");
        if (id_sort_padded_count_ != output_count_) {
            id_sort_scratch_buffer_ = create_buffer(
                    CL_MEM_READ_WRITE,
                    sizeof(int32_t) * static_cast<size_t>(id_sort_padded_count_),
                    nullptr,
                    "id_sort_scratch_buffer");
        }

        src_q_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(uint16_t) * static_cast<size_t>(k4_count_) * static_cast<size_t>(src_rows_),
                nullptr,
                "src_q_buffer");
        src_d_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(uint16_t) * static_cast<size_t>(kb_count_) * static_cast<size_t>(src_rows_),
                nullptr,
                "src_d_buffer");
        gather_ids_buffer_ = create_buffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(int32_t) * static_cast<size_t>(gather_rows_),
                gather_ids_.data(),
                "gather_ids_buffer");
        dst_q_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(uint16_t) * static_cast<size_t>(k4_count_) * static_cast<size_t>(gather_rows_),
                nullptr,
                "dst_q_buffer");
        dst_d_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(uint16_t) * static_cast<size_t>(kb_count_) * static_cast<size_t>(gather_rows_),
                nullptr,
                "dst_d_buffer");
        hidden_buffer_ = create_buffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * hidden_.size(),
                hidden_.data(),
                "hidden_buffer");
        initialize_dense_b_image();
        dense_out_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(float) * indexed_output_element_count(),
                nullptr,
                "dense_out_buffer");
        reference_out_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(float) * static_cast<size_t>(gather_rows_),
                nullptr,
                "reference_out_buffer");
        indexed_out_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(float) * indexed_output_element_count(),
                nullptr,
                "indexed_out_buffer");

        initialize_pattern_buffer(src_q_buffer_, static_cast<uint32_t>(k4_count_), static_cast<uint32_t>(src_rows_), kPatternSeed0Q, kPatternSeed1Q);
        fill_u16_buffer(src_d_buffer_, kHalfOneBits, static_cast<size_t>(kb_count_) * static_cast<size_t>(src_rows_));

        // Scratch buffers for the bucket-select top-k path.
        bucket_hist_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(uint32_t) * static_cast<size_t>(kBucketCount),
                nullptr,
                "bucket_hist_buffer");
        bucket_counters_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(uint32_t) * 2,
                nullptr,
                "bucket_counters_buffer");
        bucket_output_buffer_ = create_buffer(
                CL_MEM_READ_WRITE,
                sizeof(int32_t) * static_cast<size_t>(output_count_),
                nullptr,
                "bucket_output_buffer");
    }

    size_t indexed_output_element_count() const {
        return static_cast<size_t>(gather_rows_) * static_cast<size_t>(options_.lmhead_batch);
    }

    void initialize_dense_b_image() {
        const int dense_tile_n = static_cast<int>(round_up(static_cast<size_t>(options_.lmhead_batch), size_t(8)));
        const int hidden_dim = options_.hidden_dim;
        if ((hidden_dim % 4) != 0) {
            fail("hidden_dim must be divisible by 4 for mul_mat_Ab_Bi_8x4 image input");
        }

        std::vector<uint16_t> b_half(static_cast<size_t>(dense_tile_n) * static_cast<size_t>(hidden_dim), 0);
        for (int n = 0; n < options_.lmhead_batch; ++n) {
            for (int k = 0; k < hidden_dim; ++k) {
                const size_t offset = static_cast<size_t>(n) * static_cast<size_t>(hidden_dim) + static_cast<size_t>(k);
                b_half[offset] = float_to_half_bits(hidden_[offset]);
            }
        }

        dense_b_half_buffer_ = create_buffer(
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(uint16_t) * b_half.size(),
                b_half.data(),
                "dense_b_half_buffer");

        cl_image_format image_format{};
        image_format.image_channel_order = CL_RGBA;
        image_format.image_channel_data_type = CL_HALF_FLOAT;

        cl_image_desc image_desc{};
        image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        image_desc.image_width = static_cast<size_t>(hidden_dim) * dense_tile_n / 4;
        image_desc.buffer = dense_b_half_buffer_;

        cl_int status = CL_SUCCESS;
        dense_b_image_ = clCreateImage(context_, CL_MEM_READ_ONLY, &image_format, &image_desc, nullptr, &status);
        throw_on_cl_error(status, "clCreateImage(dense B half image)");
    }

    double run_topk_once(size_t lws) {
        throw_on_cl_error(clEnqueueCopyBuffer(queue_, baseline_scores_buffer_, working_scores_buffer_,
                                              0, 0,
                                              sizeof(float) * static_cast<size_t>(padded_score_count_),
                                              0, nullptr, nullptr),
                          "clEnqueueCopyBuffer(topk scores)");
        throw_on_cl_error(clFinish(queue_), "clFinish(topk pre-copy)");

        const int32_t n_scores = options_.n_scores;
        const int32_t padded_count = padded_score_count_;
        const int32_t pad_index = std::numeric_limits<int32_t>::max();

        throw_on_cl_error(clSetKernelArg(init_i32_range_kernel_, 0, sizeof(cl_mem), &working_indices_buffer_), "clSetKernelArg(topk init arg0)");
        throw_on_cl_error(clSetKernelArg(init_i32_range_kernel_, 1, sizeof(int32_t), &n_scores), "clSetKernelArg(topk init arg1)");
        throw_on_cl_error(clSetKernelArg(init_i32_range_kernel_, 2, sizeof(int32_t), &padded_count), "clSetKernelArg(topk init arg2)");
        throw_on_cl_error(clSetKernelArg(init_i32_range_kernel_, 3, sizeof(int32_t), &pad_index), "clSetKernelArg(topk init arg3)");

        throw_on_cl_error(clSetKernelArg(sort_f32_i32_kernel_, 0, sizeof(cl_mem), &working_scores_buffer_), "clSetKernelArg(topk sort arg0)");
        throw_on_cl_error(clSetKernelArg(sort_f32_i32_kernel_, 1, sizeof(cl_mem), &working_indices_buffer_), "clSetKernelArg(topk sort arg1)");
        throw_on_cl_error(clSetKernelArg(sort_f32_i32_kernel_, 2, sizeof(int32_t), &padded_count), "clSetKernelArg(topk sort arg2)");

        const size_t init_gws[] = { lws == 0 ? static_cast<size_t>(padded_count) : round_up(static_cast<size_t>(padded_count), lws) };
        const size_t init_lws[] = { lws };
        const size_t sort_gws[] = { lws == 0 ? static_cast<size_t>(padded_count) : round_up(static_cast<size_t>(padded_count), lws) };
        const size_t sort_lws[] = { lws };

        const auto start = std::chrono::steady_clock::now();

        throw_on_cl_error(clEnqueueNDRangeKernel(queue_, init_i32_range_kernel_, 1, nullptr,
                                                 init_gws,
                                                 lws == 0 ? nullptr : init_lws,
                                                 0, nullptr, nullptr),
                          "clEnqueueNDRangeKernel(topk init)");

        for (int32_t stage_k = 2; stage_k <= padded_count; stage_k <<= 1) {
            for (int32_t stage_j = stage_k >> 1; stage_j > 0; stage_j >>= 1) {
                throw_on_cl_error(clSetKernelArg(sort_f32_i32_kernel_, 3, sizeof(int32_t), &stage_j), "clSetKernelArg(topk sort arg3)");
                throw_on_cl_error(clSetKernelArg(sort_f32_i32_kernel_, 4, sizeof(int32_t), &stage_k), "clSetKernelArg(topk sort arg4)");
                throw_on_cl_error(clEnqueueNDRangeKernel(queue_, sort_f32_i32_kernel_, 1, nullptr,
                                                         sort_gws,
                                                         lws == 0 ? nullptr : sort_lws,
                                                         0, nullptr, nullptr),
                                  "clEnqueueNDRangeKernel(topk sort step)");
            }
        }

        throw_on_cl_error(clFinish(queue_), "clFinish(topk)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double run_bucket_topk_once(size_t lws) {
        // Refresh input scores (padded buffer already contains the raw scores
        // in the first n_scores slots from baseline_scores_buffer_).
        throw_on_cl_error(clEnqueueCopyBuffer(queue_, baseline_scores_buffer_, working_scores_buffer_,
                                              0, 0,
                                              sizeof(float) * static_cast<size_t>(options_.n_scores),
                                              0, nullptr, nullptr),
                          "clEnqueueCopyBuffer(bucket-topk scores)");
        throw_on_cl_error(clFinish(queue_), "clFinish(bucket-topk pre-copy)");

        const int32_t n_scores = options_.n_scores;
        const int32_t top_k    = output_count_;

        const auto start = std::chrono::steady_clock::now();

        // Zero histogram + counters.
        const uint32_t zero = 0u;
        throw_on_cl_error(clEnqueueFillBuffer(queue_, bucket_hist_buffer_,
                                              &zero, sizeof(uint32_t),
                                              0, sizeof(uint32_t) * static_cast<size_t>(kBucketCount),
                                              0, nullptr, nullptr),
                          "clEnqueueFillBuffer(bucket-hist)");
        throw_on_cl_error(clEnqueueFillBuffer(queue_, bucket_counters_buffer_,
                                              &zero, sizeof(uint32_t),
                                              0, sizeof(uint32_t) * 2,
                                              0, nullptr, nullptr),
                          "clEnqueueFillBuffer(bucket-counters)");

        // Pass 1: histogram.
        {
            const int32_t nb = kBucketCount;
            const int32_t bs = kBucketShift;
            throw_on_cl_error(clSetKernelArg(topk_hist_kernel_, 0, sizeof(cl_mem), &working_scores_buffer_), "clSetKernelArg(hist arg0)");
            throw_on_cl_error(clSetKernelArg(topk_hist_kernel_, 1, sizeof(int32_t), &n_scores), "clSetKernelArg(hist arg1)");
            throw_on_cl_error(clSetKernelArg(topk_hist_kernel_, 2, sizeof(cl_mem), &bucket_hist_buffer_), "clSetKernelArg(hist arg2)");
            throw_on_cl_error(clSetKernelArg(topk_hist_kernel_, 3, sizeof(int32_t), &nb), "clSetKernelArg(hist arg3)");
            throw_on_cl_error(clSetKernelArg(topk_hist_kernel_, 4, sizeof(int32_t), &bs), "clSetKernelArg(hist arg4)");

            const size_t gws[] = { lws == 0 ? static_cast<size_t>(n_scores) : round_up(static_cast<size_t>(n_scores), lws) };
            const size_t lws_arr[] = { lws };
            throw_on_cl_error(clEnqueueNDRangeKernel(queue_, topk_hist_kernel_, 1, nullptr,
                                                     gws,
                                                     lws == 0 ? nullptr : lws_arr,
                                                     0, nullptr, nullptr),
                              "clEnqueueNDRangeKernel(hist)");
        }

        // Host readback of the histogram to determine the threshold bucket.
        std::vector<uint32_t> hist(static_cast<size_t>(kBucketCount), 0u);
        throw_on_cl_error(clEnqueueReadBuffer(queue_, bucket_hist_buffer_, CL_TRUE,
                                              0, sizeof(uint32_t) * static_cast<size_t>(kBucketCount),
                                              hist.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(bucket-hist)");

        uint32_t acc = 0;
        uint32_t threshold_bucket = 0;
        uint32_t taken_above = 0;
        uint32_t quota_at = 0;
        const uint32_t kk = static_cast<uint32_t>(top_k);
        for (int b = kBucketCount - 1; b >= 0; --b) {
            if (acc + hist[b] >= kk) {
                threshold_bucket = static_cast<uint32_t>(b);
                taken_above = acc;
                quota_at = kk - acc;
                break;
            }
            acc += hist[b];
        }

        // Pass 2: compact.
        {
            const int32_t bs = kBucketShift;
            throw_on_cl_error(clSetKernelArg(topk_compact_kernel_, 0, sizeof(cl_mem),  &working_scores_buffer_), "clSetKernelArg(compact arg0)");
            throw_on_cl_error(clSetKernelArg(topk_compact_kernel_, 1, sizeof(int32_t), &n_scores), "clSetKernelArg(compact arg1)");
            throw_on_cl_error(clSetKernelArg(topk_compact_kernel_, 2, sizeof(int32_t), &bs), "clSetKernelArg(compact arg2)");
            throw_on_cl_error(clSetKernelArg(topk_compact_kernel_, 3, sizeof(uint32_t), &threshold_bucket), "clSetKernelArg(compact arg3)");
            throw_on_cl_error(clSetKernelArg(topk_compact_kernel_, 4, sizeof(uint32_t), &taken_above), "clSetKernelArg(compact arg4)");
            throw_on_cl_error(clSetKernelArg(topk_compact_kernel_, 5, sizeof(uint32_t), &quota_at), "clSetKernelArg(compact arg5)");
            throw_on_cl_error(clSetKernelArg(topk_compact_kernel_, 6, sizeof(cl_mem),  &bucket_counters_buffer_), "clSetKernelArg(compact arg6)");
            throw_on_cl_error(clSetKernelArg(topk_compact_kernel_, 7, sizeof(cl_mem),  &bucket_output_buffer_), "clSetKernelArg(compact arg7)");

            const size_t gws[] = { lws == 0 ? static_cast<size_t>(n_scores) : round_up(static_cast<size_t>(n_scores), lws) };
            const size_t lws_arr[] = { lws };
            throw_on_cl_error(clEnqueueNDRangeKernel(queue_, topk_compact_kernel_, 1, nullptr,
                                                     gws,
                                                     lws == 0 ? nullptr : lws_arr,
                                                     0, nullptr, nullptr),
                              "clEnqueueNDRangeKernel(compact)");
        }

        throw_on_cl_error(clFinish(queue_), "clFinish(bucket-topk)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    void validate_bucket_topk_output() {
        std::vector<int32_t> got(static_cast<size_t>(output_count_), -1);
        throw_on_cl_error(clEnqueueReadBuffer(queue_, bucket_output_buffer_, CL_TRUE,
                                              0, sizeof(int32_t) * static_cast<size_t>(output_count_),
                                              got.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(bucket-topk output)");

        // Exact top-k test: the returned indices (unordered) must be a subset
        // of the reference top-k. We already have topk_desc_ref_ (descending
        // by score). Use a hash set to check equality.
        std::unordered_set<int32_t> ref_set(topk_desc_ref_.begin(), topk_desc_ref_.end());
        std::unordered_set<int32_t> got_set;
        got_set.reserve(got.size());
        for (int32_t v : got) {
            if (v < 0 || v >= options_.n_scores) {
                fail("bucket-topk produced out-of-range index " + std::to_string(v));
            }
            got_set.insert(v);
        }
        if (got_set.size() != ref_set.size()) {
            fail("bucket-topk output has " + std::to_string(got_set.size())
                 + " unique indices (expected " + std::to_string(ref_set.size()) + ")");
        }
        // Count mismatches. With continuous score distributions the set should
        // match exactly; if tied boundary values exist a small diff is OK.
        size_t missing = 0;
        for (int32_t v : ref_set) {
            if (got_set.find(v) == got_set.end()) {
                ++missing;
            }
        }
        if (missing > 0) {
            // Allow for ties on the threshold bucket boundary only.
            if (missing > static_cast<size_t>(output_count_) / 64) {
                fail("bucket-topk mismatched " + std::to_string(missing) + " indices vs reference");
            }
        }
    }

    double run_id_sort_once(size_t lws) {
        throw_on_cl_error(clEnqueueCopyBuffer(queue_, baseline_ids_buffer_, working_ids_buffer_,
                                              0, 0,
                                              sizeof(int32_t) * static_cast<size_t>(output_count_),
                                              0, nullptr, nullptr),
                          "clEnqueueCopyBuffer(id-sort ids)");
        throw_on_cl_error(clFinish(queue_), "clFinish(id-sort pre-copy)");

        const int32_t padded_count = id_sort_padded_count_;
        cl_mem sort_buffer = working_ids_buffer_;

        const auto start = std::chrono::steady_clock::now();

        if (id_sort_padded_count_ != output_count_) {
            sort_buffer = id_sort_scratch_buffer_;
            const int32_t pad_value = std::numeric_limits<int32_t>::max();
            throw_on_cl_error(clSetKernelArg(fill_i32_kernel_, 0, sizeof(cl_mem), &sort_buffer), "clSetKernelArg(id-sort fill arg0)");
            throw_on_cl_error(clSetKernelArg(fill_i32_kernel_, 1, sizeof(int32_t), &padded_count), "clSetKernelArg(id-sort fill arg1)");
            throw_on_cl_error(clSetKernelArg(fill_i32_kernel_, 2, sizeof(int32_t), &pad_value), "clSetKernelArg(id-sort fill arg2)");

            const size_t fill_gws[] = { lws == 0 ? static_cast<size_t>(padded_count) : round_up(static_cast<size_t>(padded_count), lws) };
            const size_t fill_lws[] = { lws };
            throw_on_cl_error(clEnqueueNDRangeKernel(queue_, fill_i32_kernel_, 1, nullptr,
                                                     fill_gws,
                                                     lws == 0 ? nullptr : fill_lws,
                                                     0, nullptr, nullptr),
                              "clEnqueueNDRangeKernel(id-sort fill)");

            throw_on_cl_error(clEnqueueCopyBuffer(queue_, working_ids_buffer_, sort_buffer,
                                                  0, 0,
                                                  sizeof(int32_t) * static_cast<size_t>(output_count_),
                                                  0, nullptr, nullptr),
                              "clEnqueueCopyBuffer(id-sort scratch copy)");
        }

        throw_on_cl_error(clSetKernelArg(sort_i32_kernel_, 0, sizeof(cl_mem), &sort_buffer), "clSetKernelArg(id-sort arg0)");
        throw_on_cl_error(clSetKernelArg(sort_i32_kernel_, 1, sizeof(int32_t), &padded_count), "clSetKernelArg(id-sort arg1)");

        const size_t sort_gws[] = { lws == 0 ? static_cast<size_t>(padded_count) : round_up(static_cast<size_t>(padded_count), lws) };
        const size_t sort_lws[] = { lws };

        for (int32_t stage_k = 2; stage_k <= padded_count; stage_k <<= 1) {
            for (int32_t stage_j = stage_k >> 1; stage_j > 0; stage_j >>= 1) {
                throw_on_cl_error(clSetKernelArg(sort_i32_kernel_, 2, sizeof(int32_t), &stage_j), "clSetKernelArg(id-sort arg2)");
                throw_on_cl_error(clSetKernelArg(sort_i32_kernel_, 3, sizeof(int32_t), &stage_k), "clSetKernelArg(id-sort arg3)");
                throw_on_cl_error(clEnqueueNDRangeKernel(queue_, sort_i32_kernel_, 1, nullptr,
                                                         sort_gws,
                                                         lws == 0 ? nullptr : sort_lws,
                                                         0, nullptr, nullptr),
                                  "clEnqueueNDRangeKernel(id-sort sort step)");
            }
        }

        if (id_sort_padded_count_ != output_count_) {
            throw_on_cl_error(clEnqueueCopyBuffer(queue_, sort_buffer, working_ids_buffer_,
                                                  0, 0,
                                                  sizeof(int32_t) * static_cast<size_t>(output_count_),
                                                  0, nullptr, nullptr),
                              "clEnqueueCopyBuffer(id-sort output copy)");
        }

        throw_on_cl_error(clFinish(queue_), "clFinish(id-sort)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double run_legacy_gather_once(size_t lx, size_t ly) {
        const int32_t src_rows = src_rows_;
        const int32_t dst_rows = gather_rows_;
        const int32_t k4_count = k4_count_;

        throw_on_cl_error(clSetKernelArg(gather_kernel_, 0, sizeof(cl_mem), &src_q_buffer_), "clSetKernelArg(gather arg0)");
        throw_on_cl_error(clSetKernelArg(gather_kernel_, 1, sizeof(cl_mem), &src_d_buffer_), "clSetKernelArg(gather arg1)");
        throw_on_cl_error(clSetKernelArg(gather_kernel_, 2, sizeof(cl_mem), &gather_ids_buffer_), "clSetKernelArg(gather arg2)");
        throw_on_cl_error(clSetKernelArg(gather_kernel_, 3, sizeof(cl_mem), &dst_q_buffer_), "clSetKernelArg(gather arg3)");
        throw_on_cl_error(clSetKernelArg(gather_kernel_, 4, sizeof(cl_mem), &dst_d_buffer_), "clSetKernelArg(gather arg4)");
        throw_on_cl_error(clSetKernelArg(gather_kernel_, 5, sizeof(int32_t), &src_rows), "clSetKernelArg(gather arg5)");
        throw_on_cl_error(clSetKernelArg(gather_kernel_, 6, sizeof(int32_t), &dst_rows), "clSetKernelArg(gather arg6)");
        throw_on_cl_error(clSetKernelArg(gather_kernel_, 7, sizeof(int32_t), &k4_count), "clSetKernelArg(gather arg7)");

        const size_t global[] = {
            lx == 0 ? static_cast<size_t>(dst_rows) : round_up(static_cast<size_t>(dst_rows), lx),
            ly == 0 ? static_cast<size_t>(k4_count) : round_up(static_cast<size_t>(k4_count), ly),
        };
        const size_t local[] = { lx, ly };

        const auto start = std::chrono::steady_clock::now();
        throw_on_cl_error(clEnqueueNDRangeKernel(queue_, gather_kernel_, 2, nullptr,
                                                 global,
                                                 (lx == 0 || ly == 0) ? nullptr : local,
                                                 0, nullptr, nullptr),
                          "clEnqueueNDRangeKernel(gather)");
        throw_on_cl_error(clFinish(queue_), "clFinish(gather)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double run_tunable_gather_once(cl_kernel kernel, const GatherTuningConfig & config) {
        const int32_t src_rows = src_rows_;
        const int32_t dst_rows = gather_rows_;
        const int32_t k4_count = k4_count_;

        const size_t row_tile = config.lx * static_cast<size_t>(config.rows_per_thread);
        const size_t k4_tile = config.ly * static_cast<size_t>(config.k4_per_thread);
        const size_t group_count_x = (static_cast<size_t>(dst_rows) + row_tile - 1) / row_tile;
        const size_t group_count_y = (static_cast<size_t>(k4_count) + k4_tile - 1) / k4_tile;
        const size_t local_ids_bytes = config.use_local_ids
                ? row_tile * sizeof(int32_t)
                : sizeof(int32_t);

        throw_on_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_q_buffer_), "clSetKernelArg(tunable gather arg0)");
        throw_on_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_d_buffer_), "clSetKernelArg(tunable gather arg1)");
        throw_on_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &gather_ids_buffer_), "clSetKernelArg(tunable gather arg2)");
        throw_on_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_q_buffer_), "clSetKernelArg(tunable gather arg3)");
        throw_on_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_mem), &dst_d_buffer_), "clSetKernelArg(tunable gather arg4)");
        throw_on_cl_error(clSetKernelArg(kernel, 5, sizeof(int32_t), &src_rows), "clSetKernelArg(tunable gather arg5)");
        throw_on_cl_error(clSetKernelArg(kernel, 6, sizeof(int32_t), &dst_rows), "clSetKernelArg(tunable gather arg6)");
        throw_on_cl_error(clSetKernelArg(kernel, 7, sizeof(int32_t), &k4_count), "clSetKernelArg(tunable gather arg7)");
        throw_on_cl_error(clSetKernelArg(kernel, 8, local_ids_bytes, nullptr), "clSetKernelArg(tunable gather arg8)");

        const size_t global[] = {
            group_count_x * config.lx,
            group_count_y * config.ly,
        };
        const size_t local[] = { config.lx, config.ly };

        const auto start = std::chrono::steady_clock::now();
        throw_on_cl_error(clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr,
                                                 global,
                                                 local,
                                                 0, nullptr, nullptr),
                          "clEnqueueNDRangeKernel(tunable gather)");
        throw_on_cl_error(clFinish(queue_), "clFinish(tunable gather)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double run_dense_matvec_once() {
        const int32_t rows = gather_rows_;
        const int32_t n = options_.lmhead_batch;
        const int32_t padded_n = static_cast<int32_t>(round_up(static_cast<size_t>(n), size_t(8)));
        const int32_t hidden_dim = options_.hidden_dim;

        throw_on_cl_error(clSetKernelArg(production_dense_matvec_kernel_, 0, sizeof(cl_mem), &dst_q_buffer_), "clSetKernelArg(dense matvec arg0)");
        throw_on_cl_error(clSetKernelArg(production_dense_matvec_kernel_, 1, sizeof(cl_mem), &dst_d_buffer_), "clSetKernelArg(dense matvec arg1)");
        throw_on_cl_error(clSetKernelArg(production_dense_matvec_kernel_, 2, sizeof(cl_mem), &dense_b_image_), "clSetKernelArg(dense matvec arg2)");
        throw_on_cl_error(clSetKernelArg(production_dense_matvec_kernel_, 3, sizeof(cl_mem), &dense_out_buffer_), "clSetKernelArg(dense matvec arg3)");
        throw_on_cl_error(clSetKernelArg(production_dense_matvec_kernel_, 4, sizeof(int32_t), &rows), "clSetKernelArg(dense matvec arg4)");
        throw_on_cl_error(clSetKernelArg(production_dense_matvec_kernel_, 5, sizeof(int32_t), &padded_n), "clSetKernelArg(dense matvec arg5)");
        throw_on_cl_error(clSetKernelArg(production_dense_matvec_kernel_, 6, sizeof(int32_t), &hidden_dim), "clSetKernelArg(dense matvec arg6)");
        throw_on_cl_error(clSetKernelArg(production_dense_matvec_kernel_, 7, sizeof(int32_t), &n), "clSetKernelArg(dense matvec arg7)");

        size_t wi_n = 1;
        size_t wi_m = 128;
        size_t wi_k = 1;
        if (rows == 128256 && n <= 4) {
            wi_m = 32;
            wi_k = 4;
        } else if (rows == 128256) {
            wi_m = 64;
            wi_k = 2;
        } else if (rows >= 32000 && n <= 4) {
            wi_m = 32;
            wi_k = 4;
        } else if (n <= 8) {
            wi_m = 4;
            wi_k = 32;
        } else if (n <= 128) {
            wi_m = 32;
            wi_k = 4;
        } else if (n <= 512) {
            wi_m = 64;
            wi_k = 2;
        }

        constexpr size_t tile_n = 8;
        constexpr size_t tile_m = 4;
        const size_t wg_n = (static_cast<size_t>(n) + tile_n - 1) / tile_n / wi_n;
        const size_t wg_m = static_cast<size_t>(rows) / tile_m / wi_m;
        if (wg_n == 0 || wg_m == 0) {
            fail("mul_mat_Ab_Bi_8x4 work size collapsed; rows must be at least one full 4xWI_M tile");
        }

        const size_t local[] = { wi_n, wi_m, wi_k };
        const size_t global[] = { wg_n * wi_n, wg_m * wi_m, wi_k };

        const auto start = std::chrono::steady_clock::now();
        throw_on_cl_error(clEnqueueNDRangeKernel(queue_, production_dense_matvec_kernel_, 3, nullptr,
                                                 global, local, 0, nullptr, nullptr),
                          "clEnqueueNDRangeKernel(dense matvec)");
        throw_on_cl_error(clFinish(queue_), "clFinish(dense matvec)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double run_reference_dense_matvec_once() {
        const int32_t hidden_dim = options_.hidden_dim;
        const int32_t rows = gather_rows_;

        throw_on_cl_error(clSetKernelArg(dense_matvec_kernel_, 0, sizeof(cl_mem), &dst_q_buffer_), "clSetKernelArg(reference dense arg0)");
        throw_on_cl_error(clSetKernelArg(dense_matvec_kernel_, 1, sizeof(cl_mem), &dst_d_buffer_), "clSetKernelArg(reference dense arg1)");
        throw_on_cl_error(clSetKernelArg(dense_matvec_kernel_, 2, sizeof(cl_mem), &hidden_buffer_), "clSetKernelArg(reference dense arg2)");
        throw_on_cl_error(clSetKernelArg(dense_matvec_kernel_, 3, sizeof(cl_mem), &reference_out_buffer_), "clSetKernelArg(reference dense arg3)");
        throw_on_cl_error(clSetKernelArg(dense_matvec_kernel_, 4, sizeof(int32_t), &hidden_dim), "clSetKernelArg(reference dense arg4)");
        throw_on_cl_error(clSetKernelArg(dense_matvec_kernel_, 5, sizeof(int32_t), &rows), "clSetKernelArg(reference dense arg5)");

        constexpr size_t local_size = 64;
        const size_t local[] = { local_size };
        const size_t global[] = { static_cast<size_t>((rows + 7) / 8) * local_size };

        const auto start = std::chrono::steady_clock::now();
        throw_on_cl_error(clEnqueueNDRangeKernel(queue_, dense_matvec_kernel_, 1, nullptr,
                                                 global, local, 0, nullptr, nullptr),
                          "clEnqueueNDRangeKernel(reference dense)");
        throw_on_cl_error(clFinish(queue_), "clFinish(reference dense)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double run_indexed_matvec_once(cl_kernel kernel, const IndexedMatvecTuningConfig & config) {
        const int32_t hidden_dim = options_.hidden_dim;
        const int32_t src_rows = src_rows_;
        const int32_t out_rows = gather_rows_;

        throw_on_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_q_buffer_), "clSetKernelArg(indexed matvec arg0)");
        throw_on_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_d_buffer_), "clSetKernelArg(indexed matvec arg1)");
        throw_on_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &hidden_buffer_), "clSetKernelArg(indexed matvec arg2)");
        throw_on_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &gather_ids_buffer_), "clSetKernelArg(indexed matvec arg3)");
        throw_on_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_mem), &indexed_out_buffer_), "clSetKernelArg(indexed matvec arg4)");
        throw_on_cl_error(clSetKernelArg(kernel, 5, sizeof(int32_t), &hidden_dim), "clSetKernelArg(indexed matvec arg5)");
        throw_on_cl_error(clSetKernelArg(kernel, 6, sizeof(int32_t), &src_rows), "clSetKernelArg(indexed matvec arg6)");
        throw_on_cl_error(clSetKernelArg(kernel, 7, sizeof(int32_t), &out_rows), "clSetKernelArg(indexed matvec arg7)");

        const size_t subgroups_per_workgroup = std::max<size_t>(1, config.local_size / 64);
        const size_t rows_per_workgroup = static_cast<size_t>(config.rows_per_subgroup) * subgroups_per_workgroup;
        const size_t workgroups = (static_cast<size_t>(out_rows) + rows_per_workgroup - 1) / rows_per_workgroup;
        const size_t local[] = { config.local_size };
        const size_t global[] = { workgroups * config.local_size };

        const auto start = std::chrono::steady_clock::now();
        throw_on_cl_error(clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr,
                                                 global, local, 0, nullptr, nullptr),
                          "clEnqueueNDRangeKernel(indexed matvec)");
        throw_on_cl_error(clFinish(queue_), "clFinish(indexed matvec)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double run_indexed_abi_matvec_once(const IndexedMatvecTuningConfig & config) {
        const int32_t hidden_dim = options_.hidden_dim;
        const int32_t src_rows = src_rows_;
        const int32_t out_rows = gather_rows_;
        const int32_t n_rows = options_.lmhead_batch;
        cl_kernel kernel = nullptr;
        if (config.abi_local_b) {
            kernel = indexed_abi_lbtile_matvec_kernel_;
        } else if (config.abi_prefetch) {
            kernel = indexed_abi_prefetch_matvec_kernel_;
        } else if (config.abi_no_split) {
            kernel = config.abi_n_tile == 4 && config.abi_m_tile == 2
                    ? indexed_abi4_n2_nosplit_matvec_kernel_
                    : (config.abi_m_tile == 8
                    ? indexed_abi_n8m8_nosplit_matvec_kernel_
                    : (config.abi_m_tile == 1
                    ? indexed_abi_n8m1_nosplit_matvec_kernel_
                    : (config.abi_m_tile == 2
                    ? indexed_abi_n8m2_nosplit_matvec_kernel_
                    : (config.abi_n_tile == 4
                    ? indexed_abi4_nosplit_matvec_kernel_
                    : indexed_abi_nosplit_matvec_kernel_))));
        } else {
            kernel = config.abi_n_tile == 4
                    ? indexed_abi4_matvec_kernel_
                    : indexed_abi_matvec_kernel_;
        }

        throw_on_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_q_buffer_), "clSetKernelArg(indexed Ab_Bi arg0)");
        throw_on_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_d_buffer_), "clSetKernelArg(indexed Ab_Bi arg1)");
        throw_on_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dense_b_image_), "clSetKernelArg(indexed Ab_Bi arg2)");
        throw_on_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &gather_ids_buffer_), "clSetKernelArg(indexed Ab_Bi arg3)");
        throw_on_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_mem), &indexed_out_buffer_), "clSetKernelArg(indexed Ab_Bi arg4)");
        throw_on_cl_error(clSetKernelArg(kernel, 5, sizeof(int32_t), &src_rows), "clSetKernelArg(indexed Ab_Bi arg5)");
        throw_on_cl_error(clSetKernelArg(kernel, 6, sizeof(int32_t), &out_rows), "clSetKernelArg(indexed Ab_Bi arg6)");
        throw_on_cl_error(clSetKernelArg(kernel, 7, sizeof(int32_t), &hidden_dim), "clSetKernelArg(indexed Ab_Bi arg7)");
        throw_on_cl_error(clSetKernelArg(kernel, 8, sizeof(int32_t), &n_rows), "clSetKernelArg(indexed Ab_Bi arg8)");

        const size_t tile_n = static_cast<size_t>(config.abi_n_tile);
        const size_t tile_m = static_cast<size_t>(config.abi_m_tile);
        const size_t wg_n = (static_cast<size_t>(n_rows) + tile_n - 1) / tile_n;
        const size_t row_tiles = (static_cast<size_t>(out_rows) + tile_m - 1) / tile_m;
        const size_t wg_m = (row_tiles + config.wi_m - 1) / config.wi_m;
        if (wg_n == 0 || wg_m == 0) {
            fail("indexed Ab_Bi work size collapsed; rows must be at least one full 4xWI_M tile");
        }

        if (config.abi_no_split || config.abi_prefetch) {
            const size_t local[] = { 1, config.wi_m };
            const size_t global[] = { wg_n, wg_m * config.wi_m };

            const auto start = std::chrono::steady_clock::now();
            throw_on_cl_error(clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr,
                                                     global, local, 0, nullptr, nullptr),
                              "clEnqueueNDRangeKernel(indexed Ab_Bi nosplit matvec)");
            throw_on_cl_error(clFinish(queue_), "clFinish(indexed Ab_Bi nosplit matvec)");
            const auto end = std::chrono::steady_clock::now();

            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        const size_t local[] = { 1, config.wi_m, config.wi_k };
        const size_t global[] = { wg_n, wg_m * config.wi_m, config.wi_k };

        const auto start = std::chrono::steady_clock::now();
        throw_on_cl_error(clEnqueueNDRangeKernel(queue_, kernel, 3, nullptr,
                                                 global, local, 0, nullptr, nullptr),
                          "clEnqueueNDRangeKernel(indexed Ab_Bi matvec)");
        throw_on_cl_error(clFinish(queue_), "clFinish(indexed Ab_Bi matvec)");
        const auto end = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    void validate_topk_output() {
        std::vector<int32_t> actual(static_cast<size_t>(output_count_));
        throw_on_cl_error(clEnqueueReadBuffer(queue_, working_indices_buffer_, CL_TRUE,
                                              0,
                                              sizeof(int32_t) * static_cast<size_t>(output_count_),
                                              actual.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(topk validate)");

        if (actual.empty()) {
            fail("top-k validation failed: no output indices");
        }

        const float cutoff_score = padded_scores_[static_cast<size_t>(topk_desc_ref_.back())];
        std::vector<uint8_t> selected(static_cast<size_t>(options_.n_scores), 0);

        int selected_at_cutoff = 0;
        float prev_score = std::numeric_limits<float>::infinity();
        for (size_t pos = 0; pos < actual.size(); ++pos) {
            const int32_t idx = actual[pos];
            if (idx < 0 || idx >= options_.n_scores) {
                std::ostringstream oss;
                oss << "top-k validation failed: output index out of range at pos=" << pos
                    << ", idx=" << idx
                    << ", valid_range=[0," << options_.n_scores << ")";
                fail(oss.str());
            }
            if (selected[static_cast<size_t>(idx)] != 0) {
                std::ostringstream oss;
                oss << "top-k validation failed: duplicate output index at pos=" << pos
                    << ", idx=" << idx;
                fail(oss.str());
            }

            selected[static_cast<size_t>(idx)] = 1;

            const float score = padded_scores_[static_cast<size_t>(idx)];
            if (score > prev_score) {
                std::ostringstream oss;
                oss << "top-k validation failed: scores are not sorted descending at pos=" << pos
                    << ", prev_score=" << prev_score
                    << ", score=" << score;
                fail(oss.str());
            }
            if (score < cutoff_score) {
                std::ostringstream oss;
                oss << "top-k validation failed: selected score below cutoff at pos=" << pos
                    << ", score=" << score
                    << ", cutoff=" << cutoff_score;
                fail(oss.str());
            }
            if (score == cutoff_score) {
                ++selected_at_cutoff;
            }

            prev_score = score;
        }

        int required_strict = 0;
        for (int idx = 0; idx < options_.n_scores; ++idx) {
            const float score = padded_scores_[static_cast<size_t>(idx)];
            if (score > cutoff_score) {
                ++required_strict;
                if (selected[static_cast<size_t>(idx)] == 0) {
                    std::ostringstream oss;
                    oss << "top-k validation failed: missing index with score strictly above cutoff, idx="
                        << idx << ", score=" << score << ", cutoff=" << cutoff_score;
                    fail(oss.str());
                }
            }
        }

        const int expected_cutoff_count = output_count_ - required_strict;
        if (selected_at_cutoff != expected_cutoff_count) {
            std::ostringstream oss;
            oss << "top-k validation failed: unexpected number of cutoff-score selections"
                << ", selected_at_cutoff=" << selected_at_cutoff
                << ", expected=" << expected_cutoff_count
                << ", cutoff=" << cutoff_score;
            fail(oss.str());
        }
    }

    void validate_id_sort_output() {
        std::vector<int32_t> actual(static_cast<size_t>(output_count_));
        throw_on_cl_error(clEnqueueReadBuffer(queue_, working_ids_buffer_, CL_TRUE,
                                              0,
                                              sizeof(int32_t) * static_cast<size_t>(output_count_),
                                              actual.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(id-sort validate)");
        if (actual != topk_asc_ref_) {
            fail("id-sort validation failed");
        }
    }

    void validate_gather_output() {
        std::vector<uint16_t> actual_q(static_cast<size_t>(k4_count_) * static_cast<size_t>(gather_rows_));
        std::vector<uint16_t> actual_d(static_cast<size_t>(kb_count_) * static_cast<size_t>(gather_rows_));

        throw_on_cl_error(clEnqueueReadBuffer(queue_, dst_q_buffer_, CL_TRUE,
                                              0,
                                              sizeof(uint16_t) * actual_q.size(),
                                              actual_q.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(gather q validate)");
        throw_on_cl_error(clEnqueueReadBuffer(queue_, dst_d_buffer_, CL_TRUE,
                                              0,
                                              sizeof(uint16_t) * actual_d.size(),
                                              actual_d.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(gather d validate)");

        for (int dst_row = 0; dst_row < gather_rows_; ++dst_row) {
            const int src_row = gather_ids_[static_cast<size_t>(dst_row)];
            for (int k4_idx = 0; k4_idx < k4_count_; ++k4_idx) {
                const uint16_t expected = pattern_u16(static_cast<uint32_t>(k4_idx), static_cast<uint32_t>(src_row), kPatternSeed0Q, kPatternSeed1Q);
                const uint16_t actual = actual_q[static_cast<size_t>(k4_idx) * static_cast<size_t>(gather_rows_) + static_cast<size_t>(dst_row)];
                if (actual != expected) {
                    std::ostringstream oss;
                    oss << "gather q validation failed at dst_row=" << dst_row
                        << ", k4_idx=" << k4_idx
                        << ", expected=" << expected
                        << ", actual=" << actual;
                    fail(oss.str());
                }
            }

            for (int kb = 0; kb < kb_count_; ++kb) {
                const uint16_t expected = kHalfOneBits;
                const uint16_t actual = actual_d[static_cast<size_t>(kb) * static_cast<size_t>(gather_rows_) + static_cast<size_t>(dst_row)];
                if (actual != expected) {
                    std::ostringstream oss;
                    oss << "gather d validation failed at dst_row=" << dst_row
                        << ", kb=" << kb
                        << ", expected=" << expected
                        << ", actual=" << actual;
                    fail(oss.str());
                }
            }
        }
    }

    void validate_indexed_matvec_output(cl_mem expected_buffer, const char * expected_label, float tolerance, int n_rows) {
        const size_t element_count = static_cast<size_t>(gather_rows_) * static_cast<size_t>(n_rows);
        std::vector<float> reference(element_count);
        std::vector<float> indexed(element_count);

        throw_on_cl_error(clEnqueueReadBuffer(queue_, expected_buffer, CL_TRUE,
                                              0, sizeof(float) * reference.size(),
                                              reference.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(expected matvec validate)");
        throw_on_cl_error(clEnqueueReadBuffer(queue_, indexed_out_buffer_, CL_TRUE,
                                              0, sizeof(float) * indexed.size(),
                                              indexed.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(indexed matvec validate)");

        float max_abs_diff = 0.0f;
        size_t max_pos = 0;
        for (size_t i = 0; i < element_count; ++i) {
            const float lhs = reference[i];
            const float rhs = indexed[i];
            if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
                std::ostringstream oss;
                oss << "matvec validation produced non-finite output at flat_pos=" << i
                    << ", " << expected_label << "=" << lhs << ", indexed=" << rhs;
                fail(oss.str());
            }
            const float diff = std::fabs(lhs - rhs);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_pos = i;
            }
        }

        if (max_abs_diff > tolerance) {
            std::ostringstream oss;
            oss << "indexed matvec validation failed: max_abs_diff=" << max_abs_diff
                << " at flat_pos=" << max_pos
                << " vs " << expected_label;
            fail(oss.str());
        }
    }

    void validate_dense_matvec_finite() {
        std::vector<float> dense(indexed_output_element_count());
        throw_on_cl_error(clEnqueueReadBuffer(queue_, dense_out_buffer_, CL_TRUE,
                                              0, sizeof(float) * dense.size(),
                                              dense.data(),
                                              0, nullptr, nullptr),
                          "clEnqueueReadBuffer(dense matvec finite)");

        for (size_t i = 0; i < dense.size(); ++i) {
            if (!std::isfinite(dense[i])) {
                std::ostringstream oss;
                oss << "dense matvec produced non-finite output at flat_pos=" << i
                    << ", value=" << dense[i];
                fail(oss.str());
            }
        }
    }

    std::vector<size_t> make_1d_candidates(size_t kernel_limit) const {
        std::vector<size_t> candidates = { 0 };
        const size_t max_x = device_max_work_item_sizes_.empty()
                ? std::min(kernel_limit, device_max_work_group_size_)
                : device_max_work_item_sizes_[0];
        const size_t limit = std::min({ kernel_limit, device_max_work_group_size_, max_x, size_t(1024) });
        for (size_t lws : make_power_of_two_values(limit)) {
            candidates.push_back(lws);
        }
        return candidates;
    }

    std::vector<std::pair<size_t, size_t>> make_2d_candidates(size_t kernel_limit) const {
        std::vector<std::pair<size_t, size_t>> candidates = { { 0, 0 } };
        const size_t limit = std::min({ kernel_limit, device_max_work_group_size_, size_t(1024) });

        const size_t max_x = device_max_work_item_sizes_.empty() ? limit : device_max_work_item_sizes_[0];
        const size_t max_y = device_max_work_item_sizes_.size() > 1 ? device_max_work_item_sizes_[1] : 1;

        const std::vector<size_t> x_values = make_power_of_two_values(std::min(max_x, size_t(1024)));
        const std::vector<size_t> y_values = make_power_of_two_values(std::min(max_y, size_t(1024)));

        for (size_t x : x_values) {
            for (size_t y : y_values) {
                if (x * y > limit) {
                    continue;
                }
                candidates.emplace_back(x, y);
            }
        }

        return candidates;
    }

    std::vector<GatherTuningConfig> make_gather_tuning_configs() const {
        std::vector<GatherTuningConfig> configs;
        const auto wg_candidates = make_2d_candidates(device_max_work_group_size_);
        const std::vector<size_t> row_factors = make_power_of_two_values(8);
        const std::vector<size_t> k4_factors = make_power_of_two_values(4);

        for (const auto & wg : wg_candidates) {
            if (wg.first == 0 || wg.second == 0) {
                continue;
            }

            for (size_t row_factor : row_factors) {
                for (size_t k4_factor : k4_factors) {
                    for (bool use_local_ids : { false, true }) {
                        const size_t local_ids_bytes = use_local_ids
                                ? wg.first * row_factor * sizeof(int32_t)
                                : sizeof(int32_t);
                        if (local_ids_bytes > static_cast<size_t>(device_local_mem_bytes_)) {
                            continue;
                        }

                        configs.push_back({
                            wg.first,
                            wg.second,
                            static_cast<int>(row_factor),
                            static_cast<int>(k4_factor),
                            use_local_ids,
                            false,
                        });
                    }
                }
            }
        }

        return configs;
    }

    std::vector<IndexedMatvecTuningConfig> make_indexed_matvec_tuning_configs() const {
        std::vector<IndexedMatvecTuningConfig> configs;
        if (options_.lmhead_batch != 1) {
            return configs;
        }
        const size_t limit = std::min({ indexed_matvec_kernel_max_wgs_, device_max_work_group_size_, size_t(1024) });
        const std::vector<int> rows_per_subgroup_values = { 1, 2, 4, 8 };

        for (size_t local_size = 64; local_size <= limit; local_size <<= 1) {
            if (local_size % 64 != 0) {
                continue;
            }
            for (int rows_per_subgroup : rows_per_subgroup_values) {
                configs.push_back({ local_size, rows_per_subgroup });
            }
        }

        return configs;
    }

    std::vector<IndexedMatvecTuningConfig> make_indexed_abi_matvec_tuning_configs() const {
        std::vector<IndexedMatvecTuningConfig> configs;
        const std::vector<int> n_tiles = options_.lmhead_batch <= 4
                ? std::vector<int>{ 4, 8 }
                : std::vector<int>{ 8 };

        for (int n_tile : n_tiles) {
            const size_t kernel_limit = n_tile == 4
                    ? indexed_abi4_matvec_kernel_max_wgs_
                    : indexed_abi_matvec_kernel_max_wgs_;
            const size_t limit = std::min({ kernel_limit, device_max_work_group_size_, size_t(128) });
            const size_t max_m = device_max_work_item_sizes_.size() >= 2 ? device_max_work_item_sizes_[1] : limit;
            const size_t max_k = device_max_work_item_sizes_.size() >= 3 ? device_max_work_item_sizes_[2] : limit;

            for (size_t wi_k = 1; wi_k <= std::min(limit, max_k); wi_k <<= 1) {
                const size_t max_wi_m = std::min({ limit / wi_k, max_m, size_t(256) });
                const std::vector<size_t> wi_m_values = options_.allow_non_power_of_two_local
                        ? make_linear_values(max_wi_m)
                        : make_power_of_two_values(max_wi_m);
                for (size_t wi_m : wi_m_values) {
                    if (wi_m * wi_k > limit) {
                        continue;
                    }
                    configs.push_back({ wi_m * wi_k, 0, true, wi_m, wi_k, n_tile });
                }
            }
        }

        for (int n_tile : n_tiles) {
            const size_t kernel_limit = n_tile == 4
                    ? indexed_abi4_nosplit_matvec_kernel_max_wgs_
                    : indexed_abi_nosplit_matvec_kernel_max_wgs_;
            const size_t limit = std::min({ kernel_limit, device_max_work_group_size_, size_t(512) });
            const size_t max_m = device_max_work_item_sizes_.size() >= 2 ? device_max_work_item_sizes_[1] : limit;
            const size_t max_wi_m = std::min(limit, max_m);
            const std::vector<size_t> wi_m_values = options_.allow_non_power_of_two_local
                    ? make_linear_values(max_wi_m)
                    : make_power_of_two_values(max_wi_m);
            for (size_t wi_m : wi_m_values) {
                configs.push_back({ wi_m, 0, true, wi_m, 1, n_tile, true });
            }
        }

        if (std::find(n_tiles.begin(), n_tiles.end(), 4) != n_tiles.end()) {
            const size_t limit = std::min({ indexed_abi4_n2_nosplit_matvec_kernel_max_wgs_,
                                            device_max_work_group_size_,
                                            size_t(512) });
            const size_t max_m = device_max_work_item_sizes_.size() >= 2 ? device_max_work_item_sizes_[1] : limit;
            const size_t max_wi_m = std::min(limit, max_m);
            const std::vector<size_t> wi_m_values = options_.allow_non_power_of_two_local
                    ? make_linear_values(max_wi_m)
                    : make_power_of_two_values(max_wi_m);
            for (size_t wi_m : wi_m_values) {
                configs.push_back({ wi_m, 0, true, wi_m, 1, 4, true, 2 });
            }
        }

        if (std::find(n_tiles.begin(), n_tiles.end(), 8) != n_tiles.end()) {
            const size_t limit = std::min({ indexed_abi_prefetch_matvec_kernel_max_wgs_,
                                            device_max_work_group_size_,
                                            size_t(512) });
            const size_t max_m = device_max_work_item_sizes_.size() >= 2 ? device_max_work_item_sizes_[1] : limit;
            const size_t max_wi_m = std::min(limit, max_m);
            const std::vector<size_t> wi_m_values = options_.allow_non_power_of_two_local
                    ? make_linear_values(max_wi_m)
                    : make_power_of_two_values(max_wi_m);
            for (size_t wi_m : wi_m_values) {
                configs.push_back({ wi_m, 0, true, wi_m, 1, 8, true, 4, false, true });
            }
        }

        if (std::find(n_tiles.begin(), n_tiles.end(), 8) != n_tiles.end()) {
            const size_t limit = std::min({ indexed_abi_n8m1_nosplit_matvec_kernel_max_wgs_,
                                            device_max_work_group_size_,
                                            size_t(512) });
            const size_t max_m = device_max_work_item_sizes_.size() >= 2 ? device_max_work_item_sizes_[1] : limit;
            const size_t max_wi_m = std::min(limit, max_m);
            const std::vector<size_t> wi_m_values = options_.allow_non_power_of_two_local
                    ? make_linear_values(max_wi_m)
                    : make_power_of_two_values(max_wi_m);
            for (size_t wi_m : wi_m_values) {
                configs.push_back({ wi_m, 0, true, wi_m, 1, 8, true, 1 });
            }
        }

        if (std::find(n_tiles.begin(), n_tiles.end(), 8) != n_tiles.end()) {
            const size_t limit = std::min({ indexed_abi_n8m2_nosplit_matvec_kernel_max_wgs_,
                                            device_max_work_group_size_,
                                            size_t(512) });
            const size_t max_m = device_max_work_item_sizes_.size() >= 2 ? device_max_work_item_sizes_[1] : limit;
            const size_t max_wi_m = std::min(limit, max_m);
            const std::vector<size_t> wi_m_values = options_.allow_non_power_of_two_local
                    ? make_linear_values(max_wi_m)
                    : make_power_of_two_values(max_wi_m);
            for (size_t wi_m : wi_m_values) {
                configs.push_back({ wi_m, 0, true, wi_m, 1, 8, true, 2 });
            }
        }

        if (options_.lmhead_batch >= 8) {
            const size_t limit = std::min({ indexed_abi_lbtile_matvec_kernel_max_wgs_,
                                            device_max_work_group_size_,
                                            size_t(512) });
            const size_t max_m = device_max_work_item_sizes_.size() >= 2 ? device_max_work_item_sizes_[1] : limit;
            const size_t max_wi_m = std::min(limit, max_m);
            const std::vector<size_t> wi_m_values = options_.allow_non_power_of_two_local
                    ? make_linear_values(max_wi_m)
                    : make_power_of_two_values(max_wi_m);
            for (size_t wi_m : wi_m_values) {
                configs.push_back({ wi_m, 0, true, wi_m, 1, 8, true, 4, true });
            }
        }

        if (options_.lmhead_batch >= 8) {
            const size_t limit = std::min({ indexed_abi_n8m8_nosplit_matvec_kernel_max_wgs_,
                                            device_max_work_group_size_,
                                            size_t(512) });
            const size_t max_m = device_max_work_item_sizes_.size() >= 2 ? device_max_work_item_sizes_[1] : limit;
            const size_t max_wi_m = std::min(limit, max_m);
            const std::vector<size_t> wi_m_values = options_.allow_non_power_of_two_local
                    ? make_linear_values(max_wi_m)
                    : make_power_of_two_values(max_wi_m);
            for (size_t wi_m : wi_m_values) {
                configs.push_back({ wi_m, 0, true, wi_m, 1, 8, true, 8 });
            }
        }

        std::sort(configs.begin(), configs.end(), [](const IndexedMatvecTuningConfig & lhs,
                                                     const IndexedMatvecTuningConfig & rhs) {
            if (lhs.abi_n_tile != rhs.abi_n_tile) {
                return lhs.abi_n_tile < rhs.abi_n_tile;
            }
            if (lhs.abi_m_tile != rhs.abi_m_tile) {
                return lhs.abi_m_tile < rhs.abi_m_tile;
            }
            if (lhs.abi_no_split != rhs.abi_no_split) {
                return lhs.abi_no_split < rhs.abi_no_split;
            }
            if (lhs.abi_prefetch != rhs.abi_prefetch) {
                return lhs.abi_prefetch < rhs.abi_prefetch;
            }
            if (lhs.abi_local_b != rhs.abi_local_b) {
                return lhs.abi_local_b < rhs.abi_local_b;
            }
            if (lhs.wi_k != rhs.wi_k) {
                return lhs.wi_k < rhs.wi_k;
            }
            return lhs.wi_m < rhs.wi_m;
        });

        return configs;
    }

    void print_top_1d_results(const std::vector<OneDimResult> & results, const char * label, const char * current_label) const {
        std::cout << "  Best " << label << " configs:\n";
        const size_t topn = std::min<size_t>(5, results.size());
        for (size_t i = 0; i < topn; ++i) {
            std::cout << "    " << std::setw(2) << (i + 1)
                      << ". lws=" << std::setw(5) << format_lws(results[i].lws)
                      << " avg_ms=" << std::fixed << std::setprecision(3) << results[i].avg_ms
                      << " min_ms=" << results[i].min_ms << "\n";
        }

        const OneDimResult * current = nullptr;
        for (const auto & result : results) {
            if ((result.lws == 0 && std::string(current_label) == "auto") ||
                (result.lws != 0 && format_lws(result.lws) == current_label)) {
                current = &result;
                break;
            }
        }

        if (current != nullptr) {
            const double speedup = current->avg_ms / results.front().avg_ms;
            std::cout << "  Current baseline (" << current_label << ") avg_ms=" << current->avg_ms
                      << ", best speedup=" << std::fixed << std::setprecision(3) << speedup << "x\n";
        }
    }

    void print_top_2d_results(const std::vector<TwoDimResult> & results, const char * label, const char * current_label) const {
        std::cout << "  Best " << label << " configs:\n";
        const size_t topn = std::min<size_t>(5, results.size());
        for (size_t i = 0; i < topn; ++i) {
            std::cout << "    " << std::setw(2) << (i + 1)
                      << ". lws=" << std::setw(7) << format_lws(results[i].lx, results[i].ly)
                      << " avg_ms=" << std::fixed << std::setprecision(3) << results[i].avg_ms
                      << " min_ms=" << results[i].min_ms << "\n";
        }

        const TwoDimResult * current = nullptr;
        for (const auto & result : results) {
            if (format_lws(result.lx, result.ly) == current_label) {
                current = &result;
                break;
            }
        }

        if (current != nullptr) {
            const double speedup = current->avg_ms / results.front().avg_ms;
            std::cout << "  Current baseline (" << current_label << ") avg_ms=" << current->avg_ms
                      << ", best speedup=" << std::fixed << std::setprecision(3) << speedup << "x\n";
        }
    }

    void print_top_gather_results(const std::vector<GatherResult> & results) const {
        std::cout << "  Best gather configs:\n";
        const size_t topn = std::min<size_t>(5, results.size());
        for (size_t i = 0; i < topn; ++i) {
            std::cout << "    " << std::setw(2) << (i + 1)
                      << ". " << format_gather_config(results[i].config)
                      << " avg_ms=" << std::fixed << std::setprecision(3) << results[i].avg_ms
                      << " min_ms=" << results[i].min_ms << "\n";
        }

        const GatherResult * legacy = nullptr;
        for (const auto & result : results) {
            if (result.config.legacy_kernel) {
                legacy = &result;
                break;
            }
        }

        if (legacy != nullptr) {
            const double speedup = legacy->avg_ms / results.front().avg_ms;
            std::cout << "  Current baseline (" << format_gather_config(legacy->config)
                      << ") avg_ms=" << legacy->avg_ms
                      << ", best speedup=" << std::fixed << std::setprecision(3) << speedup << "x\n";
        }
    }

    std::string query_platform_string(cl_platform_id platform, cl_platform_info param) const {
        size_t size = 0;
        throw_on_cl_error(clGetPlatformInfo(platform, param, 0, nullptr, &size), "clGetPlatformInfo(size)");
        std::vector<char> buffer(size);
        throw_on_cl_error(clGetPlatformInfo(platform, param, size, buffer.data(), nullptr), "clGetPlatformInfo(value)");
        return std::string(buffer.data(), buffer.data() + size - 1);
    }

    std::string query_device_string(cl_device_id device, cl_device_info param) const {
        size_t size = 0;
        throw_on_cl_error(clGetDeviceInfo(device, param, 0, nullptr, &size), "clGetDeviceInfo(size)");
        std::vector<char> buffer(size);
        throw_on_cl_error(clGetDeviceInfo(device, param, size, buffer.data(), nullptr), "clGetDeviceInfo(value)");
        return std::string(buffer.data(), buffer.data() + size - 1);
    }

    cl_program build_program(const std::vector<std::string> & sources, const char * tag, const char * build_options = nullptr) {
        std::vector<const char *> ptrs;
        std::vector<size_t> sizes;
        ptrs.reserve(sources.size());
        sizes.reserve(sources.size());
        for (const auto & source : sources) {
            ptrs.push_back(source.c_str());
            sizes.push_back(source.size());
        }

        cl_int status = CL_SUCCESS;
        cl_program program = clCreateProgramWithSource(context_,
                                                       static_cast<cl_uint>(ptrs.size()),
                                                       ptrs.data(),
                                                       sizes.data(),
                                                       &status);
        throw_on_cl_error(status, std::string("clCreateProgramWithSource(") + tag + ")");

        status = clBuildProgram(program, 1, &device_, build_options, nullptr, nullptr);
        if (status != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size + 1, '\0');
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);

            std::ostringstream oss;
            oss << "failed to build " << tag << " kernels:\n" << log.data();
            clReleaseProgram(program);
            fail(oss.str());
        }

        return program;
    }

    GatherResult benchmark_tunable_gather_config(const GatherTuningConfig & config) {
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;

        try {
            std::ostringstream build_options;
            build_options << "-D GATHER_ROWS_PER_THREAD=" << config.rows_per_thread
                          << " -D GATHER_K4_PER_THREAD=" << config.k4_per_thread
                          << " -D GATHER_USE_LOCAL_IDS=" << (config.use_local_ids ? 1 : 0);

            program = build_program({ std::string(kTunableGatherKernels) }, "tunable-gather", build_options.str().c_str());
            kernel = create_kernel(program, "kernel_gather_rows_q4_0_tunable_i32");

            const size_t kernel_limit = query_kernel_wgs(kernel);
            if (config.lx * config.ly > kernel_limit) {
                std::ostringstream oss;
                oss << "work-group " << format_lws(config.lx, config.ly)
                    << " exceeds kernel limit " << kernel_limit;
                fail(oss.str());
            }

            const auto [avg_ms, min_ms] = benchmark_iterations(options_.warmup, options_.iters, [&]() {
                return run_tunable_gather_once(kernel, config);
            });
            validate_gather_output();

            release_cl_handle(kernel, clReleaseKernel);
            release_cl_handle(program, clReleaseProgram);
            return { config, avg_ms, min_ms };
        } catch (...) {
            release_cl_handle(kernel, clReleaseKernel);
            release_cl_handle(program, clReleaseProgram);
            throw;
        }
    }

    MatvecResult benchmark_indexed_matvec_config(const IndexedMatvecTuningConfig & config) {
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;

        try {
            std::ostringstream build_options;
            build_options << "-cl-std=CL2.0"
                          << " -cl-mad-enable -cl-unsafe-math-optimizations"
                          << " -cl-finite-math-only -cl-fast-relaxed-math"
                          << " -D INDEXED_ROWS_PER_SG=" << config.rows_per_subgroup;

            program = build_program({ std::string(kQ4MatvecKernels) }, "indexed-q4-matvec", build_options.str().c_str());
            kernel = create_kernel(program, "kernel_indexed_q4_0_matvec_8x");

            const size_t kernel_limit = query_kernel_wgs(kernel);
            if (config.local_size > kernel_limit) {
                std::ostringstream oss;
                oss << "local_size " << config.local_size
                    << " exceeds kernel limit " << kernel_limit;
                fail(oss.str());
            }

            const auto [avg_ms, min_ms] = benchmark_iterations(options_.warmup, options_.iters, [&]() {
                return run_indexed_matvec_once(kernel, config);
            });
            validate_indexed_matvec_output(reference_out_buffer_, "reference", 1e-3f, 1);

            release_cl_handle(kernel, clReleaseKernel);
            release_cl_handle(program, clReleaseProgram);
            return { "indexed-direct-tuned", config, avg_ms, min_ms };
        } catch (...) {
            release_cl_handle(kernel, clReleaseKernel);
            release_cl_handle(program, clReleaseProgram);
            throw;
        }
    }

    MatvecResult benchmark_indexed_abi_matvec_config(const IndexedMatvecTuningConfig & config) {
        const size_t kernel_limit = config.abi_local_b
                ? indexed_abi_lbtile_matvec_kernel_max_wgs_
                : (config.abi_prefetch
                ? indexed_abi_prefetch_matvec_kernel_max_wgs_
                : (config.abi_no_split
                ? (config.abi_n_tile == 4 && config.abi_m_tile == 2
                    ? indexed_abi4_n2_nosplit_matvec_kernel_max_wgs_
                    : (config.abi_m_tile == 8
                    ? indexed_abi_n8m8_nosplit_matvec_kernel_max_wgs_
                    : (config.abi_m_tile == 1
                    ? indexed_abi_n8m1_nosplit_matvec_kernel_max_wgs_
                    : (config.abi_m_tile == 2
                    ? indexed_abi_n8m2_nosplit_matvec_kernel_max_wgs_
                    : (config.abi_n_tile == 4
                    ? indexed_abi4_nosplit_matvec_kernel_max_wgs_
                    : indexed_abi_nosplit_matvec_kernel_max_wgs_)))))
                : (config.abi_n_tile == 4
                ? indexed_abi4_matvec_kernel_max_wgs_
                : indexed_abi_matvec_kernel_max_wgs_)));
        if (config.wi_m * config.wi_k > kernel_limit) {
            std::ostringstream oss;
            oss << "local_size " << (config.wi_m * config.wi_k)
                << " exceeds indexed Ab_Bi kernel limit " << kernel_limit;
            fail(oss.str());
        }

        const auto [avg_ms, min_ms] = benchmark_iterations(options_.warmup, options_.iters, [&]() {
            return run_indexed_abi_matvec_once(config);
        });
        // Different WI_K values change half-accumulation reduction order. Keep
        // the tuner permissive so we can see timing for those exploratory shapes.
        validate_indexed_matvec_output(dense_out_buffer_, "dense-Ab_Bi", 16.0f, options_.lmhead_batch);

        return { config.abi_local_b ? "indexed-Ab_Bi-8x4-localB" :
                 (config.abi_prefetch ? "indexed-Ab_Bi-8x4-nosplit-prefetch" :
                 (config.abi_no_split ? (config.abi_m_tile == 8 ? "indexed-Ab_Bi-8x8-nosplit" :
                                        (config.abi_m_tile == 1 ? "indexed-Ab_Bi-8x1-nosplit" :
                                        (config.abi_m_tile == 2 ? (config.abi_n_tile == 4 ? "indexed-Ab_Bi-4x2-nosplit" : "indexed-Ab_Bi-8x2-nosplit") :
                                        (config.abi_n_tile == 4 ? "indexed-Ab_Bi-4x4-nosplit" : "indexed-Ab_Bi-8x4-nosplit"))))
                                     : (config.abi_n_tile == 4 ? "indexed-Ab_Bi-4x4" : "indexed-Ab_Bi-8x4"))),
                 config, avg_ms, min_ms };
    }

    cl_kernel create_kernel(cl_program program, const char * name) {
        cl_int status = CL_SUCCESS;
        cl_kernel kernel = clCreateKernel(program, name, &status);
        throw_on_cl_error(status, std::string("clCreateKernel(") + name + ")");
        return kernel;
    }

    size_t query_kernel_wgs(cl_kernel kernel) const {
        size_t value = 0;
        throw_on_cl_error(clGetKernelWorkGroupInfo(kernel, device_, CL_KERNEL_WORK_GROUP_SIZE,
                                                   sizeof(value), &value, nullptr),
                          "clGetKernelWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE)");
        return value;
    }

    size_t query_kernel_preferred_multiple(cl_kernel kernel) const {
        size_t value = 0;
        throw_on_cl_error(clGetKernelWorkGroupInfo(kernel, device_, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                                   sizeof(value), &value, nullptr),
                          "clGetKernelWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)");
        return value;
    }

    cl_mem create_buffer(cl_mem_flags flags, size_t size, void * host_ptr, const char * label) {
        cl_int status = CL_SUCCESS;
        cl_mem buffer = clCreateBuffer(context_, flags, size, host_ptr, &status);
        if (status != CL_SUCCESS || buffer == nullptr) {
            std::ostringstream oss;
            oss << "failed to allocate " << label << " (" << format_bytes(size) << "): "
                << cl_status_to_string(status) << " (" << status << ")";
            fail(oss.str());
        }
        return buffer;
    }

    void initialize_pattern_buffer(cl_mem buffer, uint32_t major_dim, uint32_t minor_dim, uint32_t seed0, uint32_t seed1) {
        throw_on_cl_error(clSetKernelArg(init_pattern_kernel_, 0, sizeof(cl_mem), &buffer), "clSetKernelArg(pattern arg0)");
        throw_on_cl_error(clSetKernelArg(init_pattern_kernel_, 1, sizeof(uint32_t), &major_dim), "clSetKernelArg(pattern arg1)");
        throw_on_cl_error(clSetKernelArg(init_pattern_kernel_, 2, sizeof(uint32_t), &minor_dim), "clSetKernelArg(pattern arg2)");
        throw_on_cl_error(clSetKernelArg(init_pattern_kernel_, 3, sizeof(uint32_t), &seed0), "clSetKernelArg(pattern arg3)");
        throw_on_cl_error(clSetKernelArg(init_pattern_kernel_, 4, sizeof(uint32_t), &seed1), "clSetKernelArg(pattern arg4)");

        const size_t total = static_cast<size_t>(major_dim) * static_cast<size_t>(minor_dim);
        const size_t max_x = device_max_work_item_sizes_.empty() ? device_max_work_group_size_ : device_max_work_item_sizes_[0];
        const size_t lws = std::min<size_t>(256, std::min(device_max_work_group_size_, max_x));
        const size_t local[] = { std::max<size_t>(1, lws) };
        const size_t global[] = { round_up(total, local[0]) };

        throw_on_cl_error(clEnqueueNDRangeKernel(queue_, init_pattern_kernel_, 1, nullptr,
                                                 global, local, 0, nullptr, nullptr),
                          "clEnqueueNDRangeKernel(pattern init)");
        throw_on_cl_error(clFinish(queue_), "clFinish(pattern init)");
    }

    void fill_u16_buffer(cl_mem buffer, uint16_t value, size_t count) {
        throw_on_cl_error(clEnqueueFillBuffer(queue_, buffer,
                                              &value, sizeof(value),
                                              0, sizeof(uint16_t) * count,
                                              0, nullptr, nullptr),
                          "clEnqueueFillBuffer(u16)");
        throw_on_cl_error(clFinish(queue_), "clFinish(u16 fill)");
    }

private:
    const Options options_;
    const int padded_score_count_;
    const int output_count_;
    const int id_sort_padded_count_;
    const int gather_rows_;
    const int src_rows_;
    const int k4_count_;
    const int kb_count_;

    std::vector<float> padded_scores_;
    std::vector<int32_t> topk_desc_ref_;
    std::vector<int32_t> topk_asc_ref_;
    std::vector<int32_t> gather_ids_;
    std::vector<float> hidden_;

    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;

    std::string platform_name_;
    std::string device_name_;
    std::string kernel_source_description_;
    cl_ulong global_mem_bytes_ = 0;
    cl_ulong device_local_mem_bytes_ = 0;
    size_t device_max_work_group_size_ = 0;
    std::vector<size_t> device_max_work_item_sizes_;

    cl_program argsort_program_ = nullptr;
    cl_program gather_program_ = nullptr;
    cl_program matvec_program_ = nullptr;
    cl_program production_dense_matvec_program_ = nullptr;
    cl_kernel init_i32_range_kernel_ = nullptr;
    cl_kernel fill_i32_kernel_ = nullptr;
    cl_kernel sort_f32_i32_kernel_ = nullptr;
    cl_kernel sort_i32_kernel_ = nullptr;
    cl_kernel topk_hist_kernel_ = nullptr;
    cl_kernel topk_compact_kernel_ = nullptr;
    cl_kernel gather_kernel_ = nullptr;
    cl_kernel init_pattern_kernel_ = nullptr;
    cl_kernel dense_matvec_kernel_ = nullptr;
    cl_kernel indexed_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi4_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi_nosplit_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi4_nosplit_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi4_n2_nosplit_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi_lbtile_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi_prefetch_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi_n8m1_nosplit_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi_n8m2_nosplit_matvec_kernel_ = nullptr;
    cl_kernel indexed_abi_n8m8_nosplit_matvec_kernel_ = nullptr;
    cl_kernel production_dense_matvec_kernel_ = nullptr;

    size_t sort_f32_i32_kernel_max_wgs_ = 0;
    size_t sort_i32_kernel_max_wgs_ = 0;
    size_t topk_hist_kernel_max_wgs_ = 0;
    size_t topk_compact_kernel_max_wgs_ = 0;
    size_t gather_kernel_max_wgs_ = 0;
    size_t gather_kernel_preferred_multiple_ = 0;
    size_t dense_matvec_kernel_max_wgs_ = 0;
    size_t indexed_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi4_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi_nosplit_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi4_nosplit_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi4_n2_nosplit_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi_lbtile_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi_prefetch_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi_n8m1_nosplit_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi_n8m2_nosplit_matvec_kernel_max_wgs_ = 0;
    size_t indexed_abi_n8m8_nosplit_matvec_kernel_max_wgs_ = 0;
    size_t production_dense_matvec_kernel_max_wgs_ = 0;

    cl_mem bucket_hist_buffer_ = nullptr;
    cl_mem bucket_counters_buffer_ = nullptr;
    cl_mem bucket_output_buffer_ = nullptr;

    cl_mem baseline_scores_buffer_ = nullptr;
    cl_mem working_scores_buffer_ = nullptr;
    cl_mem working_indices_buffer_ = nullptr;

    cl_mem baseline_ids_buffer_ = nullptr;
    cl_mem working_ids_buffer_ = nullptr;
    cl_mem id_sort_scratch_buffer_ = nullptr;

    cl_mem src_q_buffer_ = nullptr;
    cl_mem src_d_buffer_ = nullptr;
    cl_mem gather_ids_buffer_ = nullptr;
    cl_mem dst_q_buffer_ = nullptr;
    cl_mem dst_d_buffer_ = nullptr;
    cl_mem hidden_buffer_ = nullptr;
    cl_mem dense_b_half_buffer_ = nullptr;
    cl_mem dense_b_image_ = nullptr;
    cl_mem dense_out_buffer_ = nullptr;
    cl_mem reference_out_buffer_ = nullptr;
    cl_mem indexed_out_buffer_ = nullptr;
};

} // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.show_help) {
            print_usage(argv[0]);
            return 0;
        }

        OpenCLTuner tuner(options);
        tuner.print_problem_summary();

        std::optional<OneDimResult> best_topk;
        std::optional<OneDimResult> best_bucket_topk;
        std::optional<OneDimResult> best_id_sort;
        std::optional<GatherResult> best_gather;
        std::optional<MatvecResult> dense_matvec;
        std::optional<MatvecResult> indexed_matvec;

        if (options.tune_topk) {
            best_topk = tuner.search_topk();
        }
        if (options.tune_bucket_topk) {
            best_bucket_topk = tuner.search_bucket_topk();
        }
        if (options.tune_id_sort) {
            best_id_sort = tuner.search_id_sort();
        }
        if (options.tune_gather) {
            best_gather = tuner.search_gather();
        }
        if (options.tune_indexed) {
            const auto results = tuner.search_indexed_matvec();
            dense_matvec = results.first;
            indexed_matvec = results.second;
        }

        std::cout << "\nRecommendation:\n";
        if (best_topk.has_value()) {
            std::cout << "  top-k init/sort lws : " << format_lws(best_topk->lws)
                      << " (avg_ms=" << std::fixed << std::setprecision(3) << best_topk->avg_ms << ")\n";
        }
        if (best_bucket_topk.has_value()) {
            std::cout << "  bucket-topk lws     : " << format_lws(best_bucket_topk->lws)
                      << " (avg_ms=" << std::fixed << std::setprecision(3) << best_bucket_topk->avg_ms << ")\n";
            if (best_topk.has_value()) {
                const double speedup = best_topk->avg_ms / best_bucket_topk->avg_ms;
                std::cout << "    vs bitonic top-k  : " << std::fixed << std::setprecision(2)
                          << speedup << "x faster\n";
            }
        }
        if (best_id_sort.has_value()) {
            std::cout << "  id-sort fill/sort lws: " << format_lws(best_id_sort->lws)
                      << " (avg_ms=" << std::fixed << std::setprecision(3) << best_id_sort->avg_ms << ")\n";
        }
        if (best_gather.has_value()) {
            std::cout << "  gather config       : " << format_gather_config(best_gather->config)
                      << " (avg_ms=" << std::fixed << std::setprecision(3) << best_gather->avg_ms << ")\n";
        }
        if (dense_matvec.has_value() && indexed_matvec.has_value()) {
            std::cout << "  indexed/dense q4 mv : " << indexed_matvec->label
                      << " " << format_indexed_config(indexed_matvec->indexed_config)
                      << " avg_ms=" << std::fixed << std::setprecision(3) << indexed_matvec->avg_ms
                      << " vs " << dense_matvec->label
                      << " avg_ms=" << dense_matvec->avg_ms
                      << " slowdown=" << (indexed_matvec->avg_ms / dense_matvec->avg_ms) << "x\n";
        }

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
