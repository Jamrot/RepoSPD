static void avc_luma_hz_qrt_4w_msa(const uint8_t *src, int32_t src_stride,
                                   uint8_t *dst, int32_t dst_stride,
                                   int32_t height, uint8_t hor_offset)
{
    uint8_t slide;
    uint32_t loop_cnt;
    v16i8 src0, src1, src2, src3;
    v8i16 res0, res1;
    v16i8 res, mask0, mask1, mask2;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[48], 16, mask0, mask1, mask2);
    slide = 2 + hor_offset;

    for (loop_cnt = (height >> 2); loop_cnt--;) {
        LD_SB4(src, src_stride, src0, src1, src2, src3);
        src += (4 * src_stride);

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
        src0 = __msa_sld_b(src0, src0, slide);
        src1 = __msa_sld_b(src1, src1, slide);
        src2 = __msa_sld_b(src2, src2, slide);
        src3 = __msa_sld_b(src3, src3, slide);
        src0 = (v16i8) __msa_insve_w((v4i32) src0, 1, (v4i32) src1);
        src1 = (v16i8) __msa_insve_w((v4i32) src2, 1, (v4i32) src3);
        src0 = (v16i8) __msa_insve_d((v2i64) src0, 1, (v2i64) src1);
        res = __msa_aver_s_b(res, src0);
        res = (v16i8) __msa_xori_b((v16u8) res, 128);

        ST4x4_UB(res, res, 0, 1, 2, 3, dst, dst_stride);
        dst += (4 * dst_stride);
    }
}
static void avc_luma_hz_qrt_8w_msa(const uint8_t *src, int32_t src_stride,
                                   uint8_t *dst, int32_t dst_stride,
                                   int32_t height, uint8_t hor_offset)
{
    uint8_t slide;
    uint32_t loop_cnt;
    v16i8 src0, src1, src2, src3;
    v16i8 tmp0, tmp1;
    v8i16 res0, res1, res2, res3;
    v16i8 mask0, mask1, mask2;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5;
    v16i8 vec6, vec7, vec8, vec9, vec10, vec11;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[0], 16, mask0, mask1, mask2);
    slide = 2 + hor_offset;

    for (loop_cnt = height >> 2; loop_cnt--;) {
        LD_SB4(src, src_stride, src0, src1, src2, src3);
        src += (4 * src_stride);

        XORI_B4_128_SB(src0, src1, src2, src3);
        VSHF_B2_SB(src0, src0, src1, src1, mask0, mask0, vec0, vec1);
        VSHF_B2_SB(src2, src2, src3, src3, mask0, mask0, vec2, vec3);
        HADD_SB4_SH(vec0, vec1, vec2, vec3, res0, res1, res2, res3);
        VSHF_B2_SB(src0, src0, src1, src1, mask1, mask1, vec4, vec5);
        VSHF_B2_SB(src2, src2, src3, src3, mask1, mask1, vec6, vec7);
        DPADD_SB4_SH(vec4, vec5, vec6, vec7, minus5b, minus5b, minus5b, minus5b,
                     res0, res1, res2, res3);
        VSHF_B2_SB(src0, src0, src1, src1, mask2, mask2, vec8, vec9);
        VSHF_B2_SB(src2, src2, src3, src3, mask2, mask2, vec10, vec11);
        DPADD_SB4_SH(vec8, vec9, vec10, vec11, plus20b, plus20b, plus20b,
                     plus20b, res0, res1, res2, res3);

        src0 = __msa_sld_b(src0, src0, slide);
        src1 = __msa_sld_b(src1, src1, slide);
        src2 = __msa_sld_b(src2, src2, slide);
        src3 = __msa_sld_b(src3, src3, slide);

        SRARI_H4_SH(res0, res1, res2, res3, 5);
        SAT_SH4_SH(res0, res1, res2, res3, 7);
        PCKEV_B2_SB(res1, res0, res3, res2, tmp0, tmp1);
        PCKEV_D2_SB(src1, src0, src3, src2, src0, src1);

        tmp0 = __msa_aver_s_b(tmp0, src0);
        tmp1 = __msa_aver_s_b(tmp1, src1);

        XORI_B2_128_SB(tmp0, tmp1);
        ST8x4_UB(tmp0, tmp1, dst, dst_stride);

        dst += (4 * dst_stride);
    }
}
static void avc_luma_hz_qrt_16w_msa(const uint8_t *src, int32_t src_stride,
                                    uint8_t *dst, int32_t dst_stride,
                                    int32_t height, uint8_t hor_offset)
{
    uint32_t loop_cnt;
    v16i8 dst0, dst1;
    v16i8 src0, src1, src2, src3;
    v16i8 mask0, mask1, mask2, vshf;
    v8i16 res0, res1, res2, res3;
    v16i8 vec0, vec1, vec2, vec3, vec4, vec5;
    v16i8 vec6, vec7, vec8, vec9, vec10, vec11;
    v16i8 minus5b = __msa_ldi_b(-5);
    v16i8 plus20b = __msa_ldi_b(20);

    LD_SB3(&luma_mask_arr[0], 16, mask0, mask1, mask2);

    if (hor_offset) {
        vshf = LD_SB(&luma_mask_arr[16 + 96]);
    } else {
        vshf = LD_SB(&luma_mask_arr[96]);
    }

    for (loop_cnt = height >> 1; loop_cnt--;) {
        LD_SB2(src, 8, src0, src1);
        src += src_stride;
        LD_SB2(src, 8, src2, src3);
        src += src_stride;

        XORI_B4_128_SB(src0, src1, src2, src3);
        VSHF_B2_SB(src0, src0, src1, src1, mask0, mask0, vec0, vec3);
        VSHF_B2_SB(src2, src2, src3, src3, mask0, mask0, vec6, vec9);
        VSHF_B2_SB(src0, src0, src1, src1, mask1, mask1, vec1, vec4);
        VSHF_B2_SB(src2, src2, src3, src3, mask1, mask1, vec7, vec10);
        VSHF_B2_SB(src0, src0, src1, src1, mask2, mask2, vec2, vec5);
        VSHF_B2_SB(src2, src2, src3, src3, mask2, mask2, vec8, vec11);
        HADD_SB4_SH(vec0, vec3, vec6, vec9, res0, res1, res2, res3);
        DPADD_SB4_SH(vec1, vec4, vec7, vec10, minus5b, minus5b, minus5b,
                     minus5b, res0, res1, res2, res3);
        DPADD_SB4_SH(vec2, vec5, vec8, vec11, plus20b, plus20b, plus20b,
                     plus20b, res0, res1, res2, res3);
        VSHF_B2_SB(src0, src1, src2, src3, vshf, vshf, src0, src2);
        SRARI_H4_SH(res0, res1, res2, res3, 5);
        SAT_SH4_SH(res0, res1, res2, res3, 7);
        PCKEV_B2_SB(res1, res0, res3, res2, dst0, dst1);

        dst0 = __msa_aver_s_b(dst0, src0);
        dst1 = __msa_aver_s_b(dst1, src2);

        XORI_B2_128_SB(dst0, dst1);

        ST_SB2(dst0, dst1, dst, dst_stride);
        dst += (2 * dst_stride);
    }
}
