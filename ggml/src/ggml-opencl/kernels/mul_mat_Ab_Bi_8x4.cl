// src0_q, src0_d, src1 are transposed as a preprocessing step
// 4-bit weights are transposed in groups of 4 (unsigned short int)
// consider weights originally "next to each other", now "on top of each other"
// each fiber computes a 8x4 tile of output elements
// using unshuffled weights

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

#define WGS 128

kernel void kernel_mul_mat_Ab_Bi_8x4(
    global const ushort *src0_q,          // Q4_0 weights   [M x K/2], 4bit packed (nibbles)
    global const half   *src0_d,          // scales         [M x (K/32)]
    __read_only  image1d_buffer_t src1,     // B: N x K row-major, pixel = half4 (K/4 픽셀)
    global float *dst,                    // C: N x M, row-major
    int ne01,                               // M
    int ne02,                               // N_padded (unused)
    int ne00,                               // K
    int ne1)                                // N
{
    const int WI_M = get_local_size(1);
    const int WI_K = get_local_size(2);
    const int gid_n = get_global_id(0);
    const int gid_m = get_group_id(1);
    const int lid_m = get_local_id(1);
    const int lid_k = get_local_id(2);
    const int lsz_m = get_local_size(1);
    const int lsz_k = get_local_size(2);

    const int M   = ne01;
    const int K   = ne00;
    const int N   = ne1;
    const int K_4 = K >> 2;

    const int out_b_idx = gid_n << 3;
    const int oc4 = (gid_m * lsz_m + lid_m) << 2;
    const int out_off = oc4 + out_b_idx * M;

    global const ushort *wptr = src0_q + oc4;
    global const half   *sptr = src0_d + oc4;

    const int b_row0_pix = out_b_idx * K_4;

    half4 acc0 = (half4)0, acc1 = (half4)0, acc2 = (half4)0, acc3 = (half4)0;
    half4 acc4 = (half4)0, acc5 = (half4)0, acc6 = (half4)0, acc7 = (half4)0;

    for (int k4 = lid_k; k4 < K_4; k4 += lsz_k) {
        // scale: (k4*4)/32 == k4/8
        const half4 sc = vload4(0, sptr + (k4 >> 3) * M);

        const int p = b_row0_pix + k4;
        const half4 in0 = read_imageh(src1, p + 0*K_4);
        const half4 in1 = read_imageh(src1, p + 1*K_4);
        const half4 in2 = read_imageh(src1, p + 2*K_4);
        const half4 in3 = read_imageh(src1, p + 3*K_4);
        const half4 in4 = read_imageh(src1, p + 4*K_4);
        const half4 in5 = read_imageh(src1, p + 5*K_4);
        const half4 in6 = read_imageh(src1, p + 6*K_4);
        const half4 in7 = read_imageh(src1, p + 7*K_4);

        const ushort4 bits4 = vload4(0, wptr + k4 * M);
        half4 w;

        // nibble j=0
        w.s0 = ((bits4.s0 & 0x000F) - 8) * sc.s0;
        w.s1 = ((bits4.s1 & 0x000F) - 8) * sc.s1;
        w.s2 = ((bits4.s2 & 0x000F) - 8) * sc.s2;
        w.s3 = ((bits4.s3 & 0x000F) - 8) * sc.s3;
        acc0 += (half4)(in0.s0) * w;  // 각 행: 열4개 동시에
        acc1 += (half4)(in1.s0) * w;
        acc2 += (half4)(in2.s0) * w;
        acc3 += (half4)(in3.s0) * w;
        acc4 += (half4)(in4.s0) * w;
        acc5 += (half4)(in5.s0) * w;
        acc6 += (half4)(in6.s0) * w;
        acc7 += (half4)(in7.s0) * w;

        // nibble j=1
        w.s0 = (((bits4.s0 & 0x00F0) >> 4) - 8) * sc.s0;
        w.s1 = (((bits4.s1 & 0x00F0) >> 4) - 8) * sc.s1;
        w.s2 = (((bits4.s2 & 0x00F0) >> 4) - 8) * sc.s2;
        w.s3 = (((bits4.s3 & 0x00F0) >> 4) - 8) * sc.s3;
        acc0 += (half4)(in0.s1) * w;
        acc1 += (half4)(in1.s1) * w;
        acc2 += (half4)(in2.s1) * w;
        acc3 += (half4)(in3.s1) * w;
        acc4 += (half4)(in4.s1) * w;
        acc5 += (half4)(in5.s1) * w;
        acc6 += (half4)(in6.s1) * w;
        acc7 += (half4)(in7.s1) * w;

        // nibble j=2
        w.s0 = (((bits4.s0 & 0x0F00) >> 8) - 8) * sc.s0;
        w.s1 = (((bits4.s1 & 0x0F00) >> 8) - 8) * sc.s1;
        w.s2 = (((bits4.s2 & 0x0F00) >> 8) - 8) * sc.s2;
        w.s3 = (((bits4.s3 & 0x0F00) >> 8) - 8) * sc.s3;
        acc0 += (half4)(in0.s2) * w;
        acc1 += (half4)(in1.s2) * w;
        acc2 += (half4)(in2.s2) * w;
        acc3 += (half4)(in3.s2) * w;
        acc4 += (half4)(in4.s2) * w;
        acc5 += (half4)(in5.s2) * w;
        acc6 += (half4)(in6.s2) * w;
        acc7 += (half4)(in7.s2) * w;

        // nibble j=3
        w.s0 = (((bits4.s0 & 0xF000) >> 12) - 8) * sc.s0;
        w.s1 = (((bits4.s1 & 0xF000) >> 12) - 8) * sc.s1;
        w.s2 = (((bits4.s2 & 0xF000) >> 12) - 8) * sc.s2;
        w.s3 = (((bits4.s3 & 0xF000) >> 12) - 8) * sc.s3;
        acc0 += (half4)(in0.s3) * w;
        acc1 += (half4)(in1.s3) * w;
        acc2 += (half4)(in2.s3) * w;
        acc3 += (half4)(in3.s3) * w;
        acc4 += (half4)(in4.s3) * w;
        acc5 += (half4)(in5.s3) * w;
        acc6 += (half4)(in6.s3) * w;
        acc7 += (half4)(in7.s3) * w;
    }

    __local half4 sum0[WGS], sum1[WGS], sum2[WGS], sum3[WGS];
    __local half4 sum4[WGS], sum5[WGS], sum6[WGS], sum7[WGS];

    if (lsz_k == 1) {
        __global float *outp = dst + out_off;
        if (out_b_idx + 0 < N) vstore4(convert_float4(acc0), 0, outp + 0*M);
        if (out_b_idx + 1 < N) vstore4(convert_float4(acc1), 0, outp + 1*M);
        if (out_b_idx + 2 < N) vstore4(convert_float4(acc2), 0, outp + 2*M);
        if (out_b_idx + 3 < N) vstore4(convert_float4(acc3), 0, outp + 3*M);
        if (out_b_idx + 4 < N) vstore4(convert_float4(acc4), 0, outp + 4*M);
        if (out_b_idx + 5 < N) vstore4(convert_float4(acc5), 0, outp + 5*M);
        if (out_b_idx + 6 < N) vstore4(convert_float4(acc6), 0, outp + 6*M);
        if (out_b_idx + 7 < N) vstore4(convert_float4(acc7), 0, outp + 7*M);
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

        if (lid_k == 0) {
            const int final_slot = lid_m * lsz_k;
            __global float *outp = dst + out_off;
            if (out_b_idx + 0 < N) vstore4(convert_float4(sum0[final_slot]), 0, outp + 0*M);
            if (out_b_idx + 1 < N) vstore4(convert_float4(sum1[final_slot]), 0, outp + 1*M);
            if (out_b_idx + 2 < N) vstore4(convert_float4(sum2[final_slot]), 0, outp + 2*M);
            if (out_b_idx + 3 < N) vstore4(convert_float4(sum3[final_slot]), 0, outp + 3*M);
            if (out_b_idx + 4 < N) vstore4(convert_float4(sum4[final_slot]), 0, outp + 4*M);
            if (out_b_idx + 5 < N) vstore4(convert_float4(sum5[final_slot]), 0, outp + 5*M);
            if (out_b_idx + 6 < N) vstore4(convert_float4(sum6[final_slot]), 0, outp + 6*M);
            if (out_b_idx + 7 < N) vstore4(convert_float4(sum7[final_slot]), 0, outp + 7*M);
        }
    }
}
