#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18
#define VTRANS_TILE_K 64
#define VTRANS_ROWS_PER_BLOCK 64

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_v_trans_1row(
        global uchar * src0_q,
        global half  * src0_d,
        ulong          src0_block_offset,
        global float * src1,
        ulong          offset1,
        global float * dst,
        ulong          offsetd,
        int            ne00,
        int            ne01,
        int            ne02,
        int            ne03,
        ulong          nb00,
        ulong          nb01,
        ulong          nb02,
        ulong          nb03,
        int            ne10,
        int            ne11,
        int            ne12,
        int            ne13,
        ulong          nb10,
        ulong          nb11,
        ulong          nb12,
        ulong          nb13,
        ulong          nb0,
        ulong          nb1,
        ulong          nb2,
        ulong          nb3,
        int            ne0,
        int            ne1,
        int            ne2,
        int            ne3,
        int            row_groups
) {
    src1 = (global float *) ((global char *) src1 + offset1);
    dst  = (global float *) ((global char *) dst  + offsetd);

    const int lane = get_local_id(0);
    const int channel_dst = get_group_id(1);

    const int sample_row_group = get_group_id(2);
    const int row_group = sample_row_group % row_groups;
    const int sample_dst = sample_row_group / row_groups;

    if (sample_dst >= ne3 || channel_dst >= ne2) {
        return;
    }

    const int channel_ratio = ne2 / ne02;
    const int sample_ratio  = ne3 / ne03;
    const int channel_x = channel_ratio > 1 ? channel_dst / channel_ratio : channel_dst;
    const int sample_x  = sample_ratio  > 1 ? sample_dst  / sample_ratio  : sample_dst;

    const int row = row_group * VTRANS_ROWS_PER_BLOCK + lane;
    const int row_block = row / QK4_0;
    const int row_in_block = row % QK4_0;
    const int q_index = row_in_block % (QK4_0 / 2);
    const int shift = row_in_block < (QK4_0 / 2) ? 0 : 4;

    local float kq_tile[VTRANS_TILE_K];

    float acc = 0.0f;

    for (int kv0 = 0; kv0 < ne00; kv0 += VTRANS_TILE_K) {
        const int kv = kv0 + lane;

        kq_tile[lane] = kv < ne00
            ? *((global float *) ((global char *) src1 + sample_dst*nb13 + channel_dst*nb12 + kv*nb10))
            : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < ne01) {
            const ulong row_base_byte =
                src0_block_offset * (ulong) Q4_0_BLOCK_SIZE +
                (ulong) sample_x * nb03 +
                (ulong) channel_x * nb02 +
                (ulong) row_block * nb01;

            for (int tk = 0; tk < VTRANS_TILE_K; ++tk) {
                const int kv_cur = kv0 + tk;
                if (kv_cur >= ne00) {
                    break;
                }

                const ulong block_byte = row_base_byte + (ulong) kv_cur * nb00;
                const ulong block_index = block_byte / (ulong) Q4_0_BLOCK_SIZE;

                const uchar q_byte = src0_q[block_index * (QK4_0 / 2) + q_index];
                const int q = ((q_byte >> shift) & 0x0F) - 8;
                acc += convert_float(src0_d[block_index]) * (float) q * kq_tile[tk];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < ne01) {
        *((global float *) ((global char *) dst + sample_dst*nb3 + channel_dst*nb2 + row*nb0)) = acc;
    }

    (void) ne10;
    (void) ne11;
    (void) ne12;
    (void) ne13;
    (void) nb11;
    (void) nb1;
    (void) ne0;
    (void) ne1;
}
