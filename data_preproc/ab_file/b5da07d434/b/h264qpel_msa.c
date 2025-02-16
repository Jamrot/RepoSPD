/*
 * Copyright (c) 2015 -2017 Parag Salasakar (Parag.Salasakar@imgtec.com)
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/mips/generic_macros_msa.h"
#include "h264dsp_mips.h"

#define AVC_CALC_DPADD_H_6PIX_2COEFF_SH(in0, in1, in2, in3, in4, in5)    \
( {                                                                      \
    v4i32 tmp0_m, tmp1_m;                                                \
    v8i16 out0_m, out1_m, out2_m, out3_m;                                \
    v8i16 minus5h_m = __msa_ldi_h(-5);                                   \
    v8i16 plus20h_m = __msa_ldi_h(20);                                   \
                                                                         \

                                                                         \
    tmp0_m = __msa_hadd_s_w((v8i16) tmp0_m, (v8i16) tmp0_m);             \
    tmp1_m = __msa_hadd_s_w((v8i16) tmp1_m, (v8i16) tmp1_m);             \
                                                                         \




                                                                         \
    SRARI_W2_SW(tmp0_m, tmp1_m, 10);                                     \
    SAT_SW2_SW(tmp0_m, tmp1_m, 7);                                       \
    out0_m = __msa_pckev_h((v8i16) tmp1_m, (v8i16) tmp0_m);              \
                                                                         \
    out0_m;                                                              \
} )

#define AVC_HORZ_FILTER_SH(in, mask0, mask1, mask2)     \
( {                                                     \
    v8i16 out0_m, out1_m;                               \
    v16i8 tmp0_m, tmp1_m;                               \
    v16i8 minus5b = __msa_ldi_b(-5);                    \
    v16i8 plus20b = __msa_ldi_b(20);                    \
                                                        \
    tmp0_m = __msa_vshf_b((v16i8) mask0, in, in);       \

                                                        \
    tmp0_m = __msa_vshf_b((v16i8) mask1, in, in);       \

                                                        \
    tmp1_m = __msa_vshf_b((v16i8) (mask2), in, in);     \

                                                        \
    out1_m;                                             \
} )

static const uint8_t luma_mask_arr[16 * 8] = {
    /* 8 width cases */
    0, 5, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 6, 11, 7, 12,
    1, 4, 2, 5, 3, 6, 4, 7, 5, 8, 6, 9, 7, 10, 8, 11,
    2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,

    /* 4 width cases */
    0, 5, 1, 6, 2, 7, 3, 8, 16, 21, 17, 22, 18, 23, 19, 24,
    1, 4, 2, 5, 3, 6, 4, 7, 17, 20, 18, 21, 19, 22, 20, 23,
    2, 3, 3, 4, 4, 5, 5, 6, 18, 19, 19, 20, 20, 21, 21, 22,

    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25,
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26
};

#define AVC_CALC_DPADD_B_6PIX_2COEFF_SH(vec0, vec1, vec2, vec3, vec4, vec5,  \
                                        out1, out2)                          \
{                                                                            \
    v16i8 tmp0_m, tmp1_m;                                                    \
    v16i8 minus5b_m = __msa_ldi_b(-5);                                       \
    v16i8 plus20b_m = __msa_ldi_b(20);                                       \
                                                                             \






}

#define AVC_CALC_DPADD_B_6PIX_2COEFF_R_SH(vec0, vec1, vec2, vec3, vec4, vec5)  \
( {                                                                            \
    v8i16 tmp1_m;                                                              \
    v16i8 tmp0_m, tmp2_m;                                                      \
    v16i8 minus5b_m = __msa_ldi_b(-5);                                         \
    v16i8 plus20b_m = __msa_ldi_b(20);                                         \
                                                                               \
    tmp1_m = (v8i16) __msa_ilvr_b((v16i8) vec5, (v16i8) vec0);                 \
    tmp1_m = __msa_hadd_s_h((v16i8) tmp1_m, (v16i8) tmp1_m);                   \
                                                                               \


                                                                               \
    tmp1_m;                                                                    \
} )

#define AVC_CALC_DPADD_H_6PIX_2COEFF_R_SH(vec0, vec1, vec2, vec3, vec4, vec5)  \
( {                                                                            \
    v4i32 tmp1_m;                                                              \
    v8i16 tmp2_m, tmp3_m;                                                      \
    v8i16 minus5h_m = __msa_ldi_h(-5);                                         \
    v8i16 plus20h_m = __msa_ldi_h(20);                                         \
                                                                               \
    tmp1_m = (v4i32) __msa_ilvr_h((v8i16) vec5, (v8i16) vec0);                 \
    tmp1_m = __msa_hadd_s_w((v8i16) tmp1_m, (v8i16) tmp1_m);                   \
                                                                               \


                                                                               \
    tmp1_m = __msa_srari_w(tmp1_m, 10);                                        \
    tmp1_m = __msa_sat_s_w(tmp1_m, 7);                                         \
                                                                               \
    tmp2_m = __msa_pckev_h((v8i16) tmp1_m, (v8i16) tmp1_m);                    \
                                                                               \
    tmp2_m;                                                                    \
} )

#define AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src0, src1,              \
                                                    mask0, mask1, mask2)     \
( {                                                                          \
    v8i16 hz_out_m;                                                          \
    v16i8 vec0_m, vec1_m, vec2_m;                                            \
    v16i8 minus5b_m = __msa_ldi_b(-5);                                       \
    v16i8 plus20b_m = __msa_ldi_b(20);                                       \
                                                                             \
    vec0_m = __msa_vshf_b((v16i8) mask0, (v16i8) src1, (v16i8) src0);        \

                                                                             \


                                                                             \
    hz_out_m;                                                                \
} )



































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































void ff_put_h264_qpel16_mc10_msa(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t stride)
{
    uint32_t loop_cnt;
    v16i8 dst0, dst1, dst2, dst3, src0, src1, src2, src3, src4, src5, src6;
    v16i8 mask0, mask1, mask2, mask3, mask4, mask5, src7, vec11;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10;
    v8i16 res0, res1, res2, res3, res4, res5, res6, res7;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[0], 16, mask0, mask1, mask2);
    mask3 = mask0 + 8;
    mask4 = mask1 + 8;
    mask5 = mask2 + 8;
    src -= 2;

    for (loop_cnt = 4; loop_cnt--;) {
        LD_SB2(src, 16, src0, src1);
        src += stride;
        LD_SB2(src, 16, src2, src3);
        src += stride;
        LD_SB2(src, 16, src4, src5);
        src += stride;
        LD_SB2(src, 16, src6, src7);
        src += stride;

        XORI_B8_128_SB(src0, src1, src2, src3, src4, src5, src6, src7);
        VSHF_B2_SB(src0, src0, src0, src1, mask0, mask3, vec0, vec3);
        VSHF_B2_SB(src2, src2, src2, src3, mask0, mask3, vec6, vec9);
        VSHF_B2_SB(src0, src0, src0, src1, mask1, mask4, vec1, vec4);
        VSHF_B2_SB(src2, src2, src2, src3, mask1, mask4, vec7, vec10);
        VSHF_B2_SB(src0, src0, src0, src1, mask2, mask5, vec2, vec5);
        VSHF_B2_SB(src2, src2, src2, src3, mask2, mask5, vec8, vec11);
        HADD_SB4_SH(vec0, vec3, vec6, vec9, res0, res1, res2, res3);
        DPADD_SB4_SH(vec1, vec4, vec7, vec10, minus5b, minus5b, minus5b,
                     minus5b, res0, res1, res2, res3);
        DPADD_SB4_SH(vec2, vec5, vec8, vec11, plus20b, plus20b, plus20b,
                     plus20b, res0, res1, res2, res3);
        VSHF_B2_SB(src4, src4, src4, src5, mask0, mask3, vec0, vec3);
        VSHF_B2_SB(src6, src6, src6, src7, mask0, mask3, vec6, vec9);
        VSHF_B2_SB(src4, src4, src4, src5, mask1, mask4, vec1, vec4);
        VSHF_B2_SB(src6, src6, src6, src7, mask1, mask4, vec7, vec10);
        VSHF_B2_SB(src4, src4, src4, src5, mask2, mask5, vec2, vec5);
        VSHF_B2_SB(src6, src6, src6, src7, mask2, mask5, vec8, vec11);
        HADD_SB4_SH(vec0, vec3, vec6, vec9, res4, res5, res6, res7);
        DPADD_SB4_SH(vec1, vec4, vec7, vec10, minus5b, minus5b, minus5b,
                     minus5b, res4, res5, res6, res7);
        DPADD_SB4_SH(vec2, vec5, vec8, vec11, plus20b, plus20b, plus20b,
                     plus20b, res4, res5, res6, res7);
        SLDI_B2_SB(src1, src3, src0, src2, src0, src2, 2);
        SLDI_B2_SB(src5, src7, src4, src6, src4, src6, 2);
        SRARI_H4_SH(res0, res1, res2, res3, 5);
        SRARI_H4_SH(res4, res5, res6, res7, 5);
        SAT_SH4_SH(res0, res1, res2, res3, 7);
        SAT_SH4_SH(res4, res5, res6, res7, 7);
        PCKEV_B2_SB(res1, res0, res3, res2, dst0, dst1);
        PCKEV_B2_SB(res5, res4, res7, res6, dst2, dst3);
        dst0 = __msa_aver_s_b(dst0, src0);
        dst1 = __msa_aver_s_b(dst1, src2);
        dst2 = __msa_aver_s_b(dst2, src4);
        dst3 = __msa_aver_s_b(dst3, src6);
        XORI_B4_128_SB(dst0, dst1, dst2, dst3);
        ST_SB4(dst0, dst1, dst2, dst3, dst, stride);
        dst += (4 * stride);
    }
}

void ff_put_h264_qpel16_mc30_msa(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t stride)
{
    uint32_t loop_cnt;
    v16i8 dst0, dst1, dst2, dst3, src0, src1, src2, src3, src4, src5, src6;
    v16i8 mask0, mask1, mask2, mask3, mask4, mask5, src7, vec11;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10;
    v8i16 res0, res1, res2, res3, res4, res5, res6, res7;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[0], 16, mask0, mask1, mask2);
    mask3 = mask0 + 8;
    mask4 = mask1 + 8;
    mask5 = mask2 + 8;
    src -= 2;

    for (loop_cnt = 4; loop_cnt--;) {
        LD_SB2(src, 16, src0, src1);
        src += stride;
        LD_SB2(src, 16, src2, src3);
        src += stride;
        LD_SB2(src, 16, src4, src5);
        src += stride;
        LD_SB2(src, 16, src6, src7);
        src += stride;

        XORI_B8_128_SB(src0, src1, src2, src3, src4, src5, src6, src7);
        VSHF_B2_SB(src0, src0, src0, src1, mask0, mask3, vec0, vec3);
        VSHF_B2_SB(src2, src2, src2, src3, mask0, mask3, vec6, vec9);
        VSHF_B2_SB(src0, src0, src0, src1, mask1, mask4, vec1, vec4);
        VSHF_B2_SB(src2, src2, src2, src3, mask1, mask4, vec7, vec10);
        VSHF_B2_SB(src0, src0, src0, src1, mask2, mask5, vec2, vec5);
        VSHF_B2_SB(src2, src2, src2, src3, mask2, mask5, vec8, vec11);
        HADD_SB4_SH(vec0, vec3, vec6, vec9, res0, res1, res2, res3);
        DPADD_SB4_SH(vec1, vec4, vec7, vec10, minus5b, minus5b, minus5b,
                     minus5b, res0, res1, res2, res3);
        DPADD_SB4_SH(vec2, vec5, vec8, vec11, plus20b, plus20b, plus20b,
                     plus20b, res0, res1, res2, res3);
        VSHF_B2_SB(src4, src4, src4, src5, mask0, mask3, vec0, vec3);
        VSHF_B2_SB(src6, src6, src6, src7, mask0, mask3, vec6, vec9);
        VSHF_B2_SB(src4, src4, src4, src5, mask1, mask4, vec1, vec4);
        VSHF_B2_SB(src6, src6, src6, src7, mask1, mask4, vec7, vec10);
        VSHF_B2_SB(src4, src4, src4, src5, mask2, mask5, vec2, vec5);
        VSHF_B2_SB(src6, src6, src6, src7, mask2, mask5, vec8, vec11);
        HADD_SB4_SH(vec0, vec3, vec6, vec9, res4, res5, res6, res7);
        DPADD_SB4_SH(vec1, vec4, vec7, vec10, minus5b, minus5b, minus5b,
                     minus5b, res4, res5, res6, res7);
        DPADD_SB4_SH(vec2, vec5, vec8, vec11, plus20b, plus20b, plus20b,
                     plus20b, res4, res5, res6, res7);
        SLDI_B2_SB(src1, src3, src0, src2, src0, src2, 3);
        SLDI_B2_SB(src5, src7, src4, src6, src4, src6, 3);
        SRARI_H4_SH(res0, res1, res2, res3, 5);
        SRARI_H4_SH(res4, res5, res6, res7, 5);
        SAT_SH4_SH(res0, res1, res2, res3, 7);
        SAT_SH4_SH(res4, res5, res6, res7, 7);
        PCKEV_B2_SB(res1, res0, res3, res2, dst0, dst1);
        PCKEV_B2_SB(res5, res4, res7, res6, dst2, dst3);
        dst0 = __msa_aver_s_b(dst0, src0);
        dst1 = __msa_aver_s_b(dst1, src2);
        dst2 = __msa_aver_s_b(dst2, src4);
        dst3 = __msa_aver_s_b(dst3, src6);
        XORI_B4_128_SB(dst0, dst1, dst2, dst3);
        ST_SB4(dst0, dst1, dst2, dst3, dst, stride);
        dst += (4 * stride);
    }
}

void ff_put_h264_qpel8_mc10_msa(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride)
{
    v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask0, mask1, mask2;
    v16i8 tmp0, tmp1, tmp2, tmp3, vec11;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10;
    v8i16 res0, res1, res2, res3, res4, res5, res6, res7;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[0], 16, mask0, mask1, mask2);
    LD_SB8(src - 2, stride, src0, src1, src2, src3, src4, src5, src6, src7);
    XORI_B8_128_SB(src0, src1, src2, src3, src4, src5, src6, src7);
    VSHF_B2_SB(src0, src0, src1, src1, mask0, mask0, vec0, vec1);
    VSHF_B2_SB(src2, src2, src3, src3, mask0, mask0, vec2, vec3);
    HADD_SB4_SH(vec0, vec1, vec2, vec3, res0, res1, res2, res3);
    VSHF_B2_SB(src0, src0, src1, src1, mask1, mask1, vec4, vec5);
    VSHF_B2_SB(src2, src2, src3, src3, mask1, mask1, vec6, vec7);
    DPADD_SB4_SH(vec4, vec5, vec6, vec7, minus5b, minus5b, minus5b, minus5b,
                 res0, res1, res2, res3);
    VSHF_B2_SB(src0, src0, src1, src1, mask2, mask2, vec8, vec9);
    VSHF_B2_SB(src2, src2, src3, src3, mask2, mask2, vec10, vec11);
    DPADD_SB4_SH(vec8, vec9, vec10, vec11, plus20b, plus20b, plus20b, plus20b,
                 res0, res1, res2, res3);
    VSHF_B2_SB(src4, src4, src5, src5, mask0, mask0, vec0, vec1);
    VSHF_B2_SB(src6, src6, src7, src7, mask0, mask0, vec2, vec3);
    HADD_SB4_SH(vec0, vec1, vec2, vec3, res4, res5, res6, res7);
    VSHF_B2_SB(src4, src4, src5, src5, mask1, mask1, vec4, vec5);
    VSHF_B2_SB(src6, src6, src7, src7, mask1, mask1, vec6, vec7);
    DPADD_SB4_SH(vec4, vec5, vec6, vec7, minus5b, minus5b, minus5b, minus5b,
                 res4, res5, res6, res7);
    VSHF_B2_SB(src4, src4, src5, src5, mask2, mask2, vec8, vec9);
    VSHF_B2_SB(src6, src6, src7, src7, mask2, mask2, vec10, vec11);
    DPADD_SB4_SH(vec8, vec9, vec10, vec11, plus20b, plus20b, plus20b, plus20b,
                 res4, res5, res6, res7);
    SLDI_B2_SB(src0, src1, src0, src1, src0, src1, 2);
    SLDI_B2_SB(src2, src3, src2, src3, src2, src3, 2);
    SLDI_B2_SB(src4, src5, src4, src5, src4, src5, 2);
    SLDI_B2_SB(src6, src7, src6, src7, src6, src7, 2);
    PCKEV_D2_SB(src1, src0, src3, src2, src0, src1);
    PCKEV_D2_SB(src5, src4, src7, src6, src4, src5);
    SRARI_H4_SH(res0, res1, res2, res3, 5);
    SRARI_H4_SH(res4, res5, res6, res7, 5);
    SAT_SH4_SH(res0, res1, res2, res3, 7);
    SAT_SH4_SH(res4, res5, res6, res7, 7);
    PCKEV_B2_SB(res1, res0, res3, res2, tmp0, tmp1);
    PCKEV_B2_SB(res5, res4, res7, res6, tmp2, tmp3);
    tmp0 = __msa_aver_s_b(tmp0, src0);
    tmp1 = __msa_aver_s_b(tmp1, src1);
    tmp2 = __msa_aver_s_b(tmp2, src4);
    tmp3 = __msa_aver_s_b(tmp3, src5);
    XORI_B4_128_SB(tmp0, tmp1, tmp2, tmp3);
    ST8x8_UB(tmp0, tmp1, tmp2, tmp3, dst, stride);
}

void ff_put_h264_qpel8_mc30_msa(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride)
{
    v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask0, mask1, mask2;
    v16i8 tmp0, tmp1, tmp2, tmp3, vec11;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10;
    v8i16 res0, res1, res2, res3, res4, res5, res6, res7;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[0], 16, mask0, mask1, mask2);
    LD_SB8(src - 2, stride, src0, src1, src2, src3, src4, src5, src6, src7);
    XORI_B8_128_SB(src0, src1, src2, src3, src4, src5, src6, src7);
    VSHF_B2_SB(src0, src0, src1, src1, mask0, mask0, vec0, vec1);
    VSHF_B2_SB(src2, src2, src3, src3, mask0, mask0, vec2, vec3);
    HADD_SB4_SH(vec0, vec1, vec2, vec3, res0, res1, res2, res3);
    VSHF_B2_SB(src0, src0, src1, src1, mask1, mask1, vec4, vec5);
    VSHF_B2_SB(src2, src2, src3, src3, mask1, mask1, vec6, vec7);
    DPADD_SB4_SH(vec4, vec5, vec6, vec7, minus5b, minus5b, minus5b, minus5b,
                 res0, res1, res2, res3);
    VSHF_B2_SB(src0, src0, src1, src1, mask2, mask2, vec8, vec9);
    VSHF_B2_SB(src2, src2, src3, src3, mask2, mask2, vec10, vec11);
    DPADD_SB4_SH(vec8, vec9, vec10, vec11, plus20b, plus20b, plus20b, plus20b,
                 res0, res1, res2, res3);
    VSHF_B2_SB(src4, src4, src5, src5, mask0, mask0, vec0, vec1);
    VSHF_B2_SB(src6, src6, src7, src7, mask0, mask0, vec2, vec3);
    HADD_SB4_SH(vec0, vec1, vec2, vec3, res4, res5, res6, res7);
    VSHF_B2_SB(src4, src4, src5, src5, mask1, mask1, vec4, vec5);
    VSHF_B2_SB(src6, src6, src7, src7, mask1, mask1, vec6, vec7);
    DPADD_SB4_SH(vec4, vec5, vec6, vec7, minus5b, minus5b, minus5b, minus5b,
                 res4, res5, res6, res7);
    VSHF_B2_SB(src4, src4, src5, src5, mask2, mask2, vec8, vec9);
    VSHF_B2_SB(src6, src6, src7, src7, mask2, mask2, vec10, vec11);
    DPADD_SB4_SH(vec8, vec9, vec10, vec11, plus20b, plus20b, plus20b, plus20b,
                 res4, res5, res6, res7);
    SLDI_B2_SB(src0, src1, src0, src1, src0, src1, 3);
    SLDI_B2_SB(src2, src3, src2, src3, src2, src3, 3);
    SLDI_B2_SB(src4, src5, src4, src5, src4, src5, 3);
    SLDI_B2_SB(src6, src7, src6, src7, src6, src7, 3);
    PCKEV_D2_SB(src1, src0, src3, src2, src0, src1);
    PCKEV_D2_SB(src5, src4, src7, src6, src4, src5);
    SRARI_H4_SH(res0, res1, res2, res3, 5);
    SRARI_H4_SH(res4, res5, res6, res7, 5);
    SAT_SH4_SH(res0, res1, res2, res3, 7);
    SAT_SH4_SH(res4, res5, res6, res7, 7);
    PCKEV_B2_SB(res1, res0, res3, res2, tmp0, tmp1);
    PCKEV_B2_SB(res5, res4, res7, res6, tmp2, tmp3);
    tmp0 = __msa_aver_s_b(tmp0, src0);
    tmp1 = __msa_aver_s_b(tmp1, src1);
    tmp2 = __msa_aver_s_b(tmp2, src4);
    tmp3 = __msa_aver_s_b(tmp3, src5);
    XORI_B4_128_SB(tmp0, tmp1, tmp2, tmp3);
    ST8x8_UB(tmp0, tmp1, tmp2, tmp3, dst, stride);
}

void ff_put_h264_qpel4_mc10_msa(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride)
{
    v16i8 src0, src1, src2, src3, res, mask0, mask1, mask2;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5;
    v8i16 res0, res1;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[48], 16, mask0, mask1, mask2);
    LD_SB4(src - 2, stride, src0, src1, src2, src3);
    XORI_B4_128_SB(src0, src1, src2, src3);
    VSHF_B2_SB(src0, src1, src2, src3, mask0, mask0, vec0, vec1);
    HADD_SB2_SH(vec0, vec1, res0, res1);
    VSHF_B2_SB(src0, src1, src2, src3, mask1, mask1, vec2, vec3);
    DPADD_SB2_SH(vec2, vec3, minus5b, minus5b, res0, res1);
    VSHF_B2_SB(src0, src1, src2, src3, mask2, mask2, vec4, vec5);
    DPADD_SB2_SH(vec4, vec5, plus20b, plus20b, res0, res1);
    SRARI_H2_SH(res0, res1, 5);
    SAT_SH2_SH(res0, res1, 7);
    res = __msa_pckev_b((v16i8) res1, (v16i8) res0);
    SLDI_B2_SB(src0, src1, src0, src1, src0, src1, 2);
    SLDI_B2_SB(src2, src3, src2, src3, src2, src3, 2);
    src0 = (v16i8) __msa_insve_w((v4i32) src0, 1, (v4i32) src1);
    src1 = (v16i8) __msa_insve_w((v4i32) src2, 1, (v4i32) src3);
    src0 = (v16i8) __msa_insve_d((v2i64) src0, 1, (v2i64) src1);
    res = __msa_aver_s_b(res, src0);
    res = (v16i8) __msa_xori_b((v16u8) res, 128);
    ST4x4_UB(res, res, 0, 1, 2, 3, dst, stride);
}

void ff_put_h264_qpel4_mc30_msa(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride)
{
    v16i8 src0, src1, src2, src3, res, mask0, mask1, mask2;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5;
    v8i16 res0, res1;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[48], 16, mask0, mask1, mask2);
    LD_SB4(src - 2, stride, src0, src1, src2, src3);
    XORI_B4_128_SB(src0, src1, src2, src3);
    VSHF_B2_SB(src0, src1, src2, src3, mask0, mask0, vec0, vec1);
    HADD_SB2_SH(vec0, vec1, res0, res1);
    VSHF_B2_SB(src0, src1, src2, src3, mask1, mask1, vec2, vec3);
    DPADD_SB2_SH(vec2, vec3, minus5b, minus5b, res0, res1);
    VSHF_B2_SB(src0, src1, src2, src3, mask2, mask2, vec4, vec5);
    DPADD_SB2_SH(vec4, vec5, plus20b, plus20b, res0, res1);
    SRARI_H2_SH(res0, res1, 5);
    SAT_SH2_SH(res0, res1, 7);
    res = __msa_pckev_b((v16i8) res1, (v16i8) res0);
    SLDI_B2_SB(src0, src1, src0, src1, src0, src1, 3);
    SLDI_B2_SB(src2, src3, src2, src3, src2, src3, 3);
    src0 = (v16i8) __msa_insve_w((v4i32) src0, 1, (v4i32) src1);
    src1 = (v16i8) __msa_insve_w((v4i32) src2, 1, (v4i32) src3);
    src0 = (v16i8) __msa_insve_d((v2i64) src0, 1, (v2i64) src1);
    res = __msa_aver_s_b(res, src0);
    res = (v16i8) __msa_xori_b((v16u8) res, 128);
    ST4x4_UB(res, res, 0, 1, 2, 3, dst, stride);
}





























































































































































































































































































































































































































































































































































































