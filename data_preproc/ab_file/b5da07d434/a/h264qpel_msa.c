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
    avc_luma_hz_qrt_16w_msa(src - 2, stride, dst, stride, 16, 0);






























































}

void ff_put_h264_qpel16_mc30_msa(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t stride)
{
    avc_luma_hz_qrt_16w_msa(src - 2, stride, dst, stride, 16, 1);






























































}

void ff_put_h264_qpel8_mc10_msa(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride)
{
    avc_luma_hz_qrt_8w_msa(src - 2, stride, dst, stride, 8, 0);

















































}

void ff_put_h264_qpel8_mc30_msa(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride)
{
    avc_luma_hz_qrt_8w_msa(src - 2, stride, dst, stride, 8, 1);

















































}

void ff_put_h264_qpel4_mc10_msa(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride)
{
    avc_luma_hz_qrt_4w_msa(src - 2, stride, dst, stride, 4, 0);

























}

void ff_put_h264_qpel4_mc30_msa(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride)
{
    avc_luma_hz_qrt_4w_msa(src - 2, stride, dst, stride, 4, 1);

























}





























































































































































































































































































































































































































































































































































































