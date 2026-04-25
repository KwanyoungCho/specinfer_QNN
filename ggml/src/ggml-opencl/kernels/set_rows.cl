#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

// v = { mp, L, d }
inline uint fastdiv(uint n, uint4 v) {
    uint msbs;
    msbs = mul_hi(n, v.s0);
    return (msbs + n) >> v.s1;
}
inline uint fastmod(uint n, uint4 v) {
    uint q = fastdiv(n, v);
    return n - q * v.s2;
}

inline void quantize_q4_0_block(
        global float * src,
        global uchar * dst_q,
        global half  * dst_d,
        ulong          block_index) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = src[j];
        const float av = fabs(v);
        if (av > amax) {
            amax = av;
            vmax = v;
        }
    }

    const float d  = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    dst_d[block_index] = convert_half(d);

    global uchar * q = dst_q + block_index * (QK4_0 / 2);
    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[j]             * id;
        const float x1 = src[QK4_0/2 + j]   * id;

        const uchar xi0 = min((uchar) 15, convert_uchar_sat_rte(x0 + 8.5f));
        const uchar xi1 = min((uchar) 15, convert_uchar_sat_rte(x1 + 8.5f));

        q[j] = xi0 | (xi1 << 4);
    }
}

#define DEFINE_SET_ROWS_Q4_0_KERNEL(KERNEL_NAME, IDX_T) \
kernel void KERNEL_NAME( \
        global char * src0, \
        ulong         offset0, \
        global char * src1, \
        ulong         offset1, \
        global uchar * dst_q, \
        global half  * dst_d, \
        ulong         dst_offset_blocks, \
        int           ne01, \
        ulong         nb01, \
        ulong         nb02, \
        ulong         nb03, \
        uint4         ne11, \
        uint4         ne12, \
        ulong         nb10, \
        ulong         nb11, \
        ulong         nb12, \
        int           nblk0, \
        ulong         nb1, \
        ulong         nb2, \
        ulong         nb3 \
) { \
    src0 = src0 + offset0; \
    src1 = src1 + offset1; \
    int i03 = get_group_id(2); \
    int i02 = get_group_id(1); \
    int i01 = get_group_id(0) * get_local_size(1) + get_local_id(1); \
    if (i01 >= ne01) { \
        return; \
    } \
    int i12 = fastmod(i03, ne12); \
    int i11 = fastmod(i02, ne11); \
    int i10 = i01; \
    IDX_T i1 = ((global IDX_T *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0]; \
    ulong dst_row_base = dst_offset_blocks + ((ulong) i1 * nb1 + (ulong) i02 * nb2 + (ulong) i03 * nb3) / (ulong) Q4_0_BLOCK_SIZE; \
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03); \
    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) { \
        quantize_q4_0_block(src_row + ind*QK4_0, dst_q, dst_d, dst_row_base + (ulong) ind); \
    } \
}

DEFINE_SET_ROWS_Q4_0_KERNEL(kernel_set_rows_q4_0_i64, long)
DEFINE_SET_ROWS_Q4_0_KERNEL(kernel_set_rows_q4_0_i32, int)

kernel void kernel_gather_rows_q4_0_transposed_i32(
        global const ushort * src_q,
        global const half   * src_d,
        global const int    * src_ids,
        global ushort       * dst_q,
        global half         * dst_d,
        int                   src_rows,
        int                   dst_rows,
        int                   k4_count
) {
    const int dst_row = get_global_id(0);
    const int k4_idx  = get_global_id(1);

    if (dst_row >= dst_rows || k4_idx >= k4_count) {
        return;
    }

    const int src_row = src_ids[dst_row];
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

kernel void kernel_gather_rows_q4_0_transposed_i32_padded(
        global const ushort * src_q,
        global const half   * src_d,
        global const int    * src_ids,
        global ushort       * dst_q,
        global half         * dst_d,
        int                   src_rows,
        int                   selected_rows,
        int                   dst_rows,
        int                   k4_count,
        int                   pad_row
) {
    const int dst_row = get_global_id(0);
    const int k4_idx  = get_global_id(1);

    if (dst_row >= dst_rows || k4_idx >= k4_count) {
        return;
    }

    const int src_row = dst_row < selected_rows ? src_ids[dst_row] : pad_row;
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

kernel void kernel_set_rows_f32_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_set_rows_f32_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}
