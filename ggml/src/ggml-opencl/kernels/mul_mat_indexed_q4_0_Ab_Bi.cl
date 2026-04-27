// Indexed Q4_0 LM-head matmul for the Adreno SOA layout.
// It mirrors mul_mat_Ab_Bi_8x4 but reads rows through ids[], avoiding a
// separate Q4_0 gather when only logits are needed.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define INDEXED_AB_BI_IF_HI_N if (compute_hi_n)
#define INDEXED_AB_BI_STORE_HI(row_idx, acc_value, out_ptr) \
    if (out_b_idx + (row_idx) < n_rows) { \
        vstore4(convert_float4(acc_value), 0, (out_ptr) + (row_idx)*out_rows); \
    }

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_indexed_q4_0_Ab_Bi_8x4_nosplit(
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
    const int compute_hi_n = n_rows > out_b_idx + 4;

    const int row0 = out4 + 0 < out_rows ? ids[out4 + 0] : 0;
    const int row1 = out4 + 1 < out_rows ? ids[out4 + 1] : 0;
    const int row2 = out4 + 2 < out_rows ? ids[out4 + 2] : 0;
    const int row3 = out4 + 3 < out_rows ? ids[out4 + 3] : 0;

    const int b_row0_pix = out_b_idx * k4_count;

    half4 acc0 = (half4)0, acc1 = (half4)0, acc2 = (half4)0, acc3 = (half4)0;
    half4 acc4 = (half4)0, acc5 = (half4)0, acc6 = (half4)0, acc7 = (half4)0;

#define INDEXED_AB_BI_8X4_ACCUM_K4(k4_value, sc_value) do { \
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
            INDEXED_AB_BI_8X4_ACCUM_K4(k4_base + kk, sc);
        }
    }

#undef INDEXED_AB_BI_8X4_ACCUM_K4

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

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_indexed_q4_0_Ab_Bi_8x2_nosplit(
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
    const int compute_hi_n = n_rows > out_b_idx + 4;

    const int row0 = out2 + 0 < out_rows ? ids[out2 + 0] : 0;
    const int row1 = out2 + 1 < out_rows ? ids[out2 + 1] : 0;

    const int b_row0_pix = out_b_idx * k4_count;

    half2 acc0 = (half2)0, acc1 = (half2)0, acc2 = (half2)0, acc3 = (half2)0;
    half2 acc4 = (half2)0, acc5 = (half2)0, acc6 = (half2)0, acc7 = (half2)0;

#define INDEXED_AB_BI_8X2_ACCUM_K4(k4_value, sc_value) do { \
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
            INDEXED_AB_BI_8X2_ACCUM_K4(k4_base + kk, sc);
        }
    }

#undef INDEXED_AB_BI_8X2_ACCUM_K4

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

#undef INDEXED_AB_BI_STORE_HI
#undef INDEXED_AB_BI_IF_HI_N
