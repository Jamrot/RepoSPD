static void copy_width16_msa(const uint8_t *src, int32_t src_stride,
                             uint8_t *dst, int32_t dst_stride,
                             int32_t height)
{
    int32_t cnt;
    v16u8 src0, src1, src2, src3, src4, src5, src6, src7;

    if (8 == height) {
        LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
        ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
    } else if (16 == height) {
        LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
        src += (8 * src_stride);
        ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
        dst += (8 * dst_stride);
        LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
        src += (8 * src_stride);
        ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
        dst += (8 * dst_stride);
    } else if (32 == height) {
        LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
        src += (8 * src_stride);
        ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
        dst += (8 * dst_stride);
        LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
        src += (8 * src_stride);
        ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
        dst += (8 * dst_stride);
        LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
        src += (8 * src_stride);
        ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
        dst += (8 * dst_stride);
        LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
        ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
    } else if (0 == height % 4) {
        for (cnt = (height >> 2); cnt--;) {
            LD_UB4(src, src_stride, src0, src1, src2, src3);
            src += (4 * src_stride);
            ST_UB4(src0, src1, src2, src3, dst, dst_stride);
            dst += (4 * dst_stride);
        }
    }
}
static void copy_width8_msa(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride,
                            int32_t height)
{
    int32_t cnt;
    uint64_t out0, out1, out2, out3, out4, out5, out6, out7;

    if (0 == height % 8) {
        for (cnt = height >> 3; cnt--;) {
            LD4(src, src_stride, out0, out1, out2, out3);
            src += (4 * src_stride);
            LD4(src, src_stride, out4, out5, out6, out7);
            src += (4 * src_stride);

            SD4(out0, out1, out2, out3, dst, dst_stride);
            dst += (4 * dst_stride);
            SD4(out4, out5, out6, out7, dst, dst_stride);
            dst += (4 * dst_stride);
        }
    } else if (0 == height % 4) {
        for (cnt = (height / 4); cnt--;) {
            LD4(src, src_stride, out0, out1, out2, out3);
            src += (4 * src_stride);

            SD4(out0, out1, out2, out3, dst, dst_stride);
            dst += (4 * dst_stride);
        }
    }
}
static void avg_width4_msa(const uint8_t *src, int32_t src_stride,
                           uint8_t *dst, int32_t dst_stride,
                           int32_t height)
{
    uint32_t tp0, tp1, tp2, tp3;
    v16u8 src0 = { 0 }, src1 = { 0 }, dst0 = { 0 }, dst1 = { 0 };

    if (8 == height) {
        LW4(src, src_stride, tp0, tp1, tp2, tp3);
        src += 4 * src_stride;
        INSERT_W4_UB(tp0, tp1, tp2, tp3, src0);
        LW4(src, src_stride, tp0, tp1, tp2, tp3);
        INSERT_W4_UB(tp0, tp1, tp2, tp3, src1);
        LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
        INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
        LW4(dst + 4 * dst_stride, dst_stride, tp0, tp1, tp2, tp3);
        INSERT_W4_UB(tp0, tp1, tp2, tp3, dst1);
        AVER_UB2_UB(src0, dst0, src1, dst1, dst0, dst1);
        ST4x8_UB(dst0, dst1, dst, dst_stride);
    } else if (4 == height) {
        LW4(src, src_stride, tp0, tp1, tp2, tp3);
        INSERT_W4_UB(tp0, tp1, tp2, tp3, src0);
        LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
        INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
        dst0 = __msa_aver_u_b(src0, dst0);
        ST4x4_UB(dst0, dst0, 0, 1, 2, 3, dst, dst_stride);
    }
}
static void avg_width8_msa(const uint8_t *src, int32_t src_stride,
                           uint8_t *dst, int32_t dst_stride,
                           int32_t height)
{
    int32_t cnt;
    uint64_t tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7;
    v16u8 src0, src1, src2, src3;
    v16u8 dst0, dst1, dst2, dst3;

    if (0 == (height % 8)) {
        for (cnt = (height >> 3); cnt--;) {
            LD4(src, src_stride, tp0, tp1, tp2, tp3);
            src += 4 * src_stride;
            LD4(src, src_stride, tp4, tp5, tp6, tp7);
            src += 4 * src_stride;
            INSERT_D2_UB(tp0, tp1, src0);
            INSERT_D2_UB(tp2, tp3, src1);
            INSERT_D2_UB(tp4, tp5, src2);
            INSERT_D2_UB(tp6, tp7, src3);
            LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
            LD4(dst + 4 * dst_stride, dst_stride, tp4, tp5, tp6, tp7);
            INSERT_D2_UB(tp0, tp1, dst0);
            INSERT_D2_UB(tp2, tp3, dst1);
            INSERT_D2_UB(tp4, tp5, dst2);
            INSERT_D2_UB(tp6, tp7, dst3);
            AVER_UB4_UB(src0, dst0, src1, dst1, src2, dst2, src3, dst3, dst0,
                        dst1, dst2, dst3);
            ST8x8_UB(dst0, dst1, dst2, dst3, dst, dst_stride);
            dst += 8 * dst_stride;
        }
    } else if (4 == height) {
        LD4(src, src_stride, tp0, tp1, tp2, tp3);
        INSERT_D2_UB(tp0, tp1, src0);
        INSERT_D2_UB(tp2, tp3, src1);
        LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
        INSERT_D2_UB(tp0, tp1, dst0);
        INSERT_D2_UB(tp2, tp3, dst1);
        AVER_UB2_UB(src0, dst0, src1, dst1, dst0, dst1);
        ST8x4_UB(dst0, dst1, dst, dst_stride);
    }
}
static void avg_width16_msa(const uint8_t *src, int32_t src_stride,
                            uint8_t *dst, int32_t dst_stride,
                            int32_t height)
{
    int32_t cnt;
    v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
    v16u8 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;

    if (0 == (height % 8)) {
        for (cnt = (height / 8); cnt--;) {
            LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
            src += (8 * src_stride);
            LD_UB8(dst, dst_stride, dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7);

            AVER_UB4_UB(src0, dst0, src1, dst1, src2, dst2, src3, dst3,
                        dst0, dst1, dst2, dst3);
            AVER_UB4_UB(src4, dst4, src5, dst5, src6, dst6, src7, dst7,
                        dst4, dst5, dst6, dst7);
            ST_UB8(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, dst, dst_stride);
            dst += (8 * dst_stride);
        }
    } else if (0 == (height % 4)) {
        for (cnt = (height / 4); cnt--;) {
            LD_UB4(src, src_stride, src0, src1, src2, src3);
            src += (4 * src_stride);
            LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

            AVER_UB4_UB(src0, dst0, src1, dst1, src2, dst2, src3, dst3,
                        dst0, dst1, dst2, dst3);
            ST_UB4(dst0, dst1, dst2, dst3, dst, dst_stride);
            dst += (4 * dst_stride);
        }
    }
}
