/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "libavcodec/aacpsdsp.h"

#include "checkasm.h"

#define N 32
#define STRIDE 128
#define BUF_SIZE (N * STRIDE)

#define randomize(buf, len) do {                                \
    int i;                                                      \
    for (i = 0; i < len; i++) {                                 \
        const INTFLOAT f = (INTFLOAT)rnd() / UINT_MAX;          \
        (buf)[i] = f;                                           \
    }                                                           \
} while (0)

#define EPS 0.005

static void test_add_squares(void)
{
    LOCAL_ALIGNED_16(INTFLOAT, dst0, [BUF_SIZE]);
    LOCAL_ALIGNED_16(INTFLOAT, dst1, [BUF_SIZE]);
    LOCAL_ALIGNED_16(INTFLOAT, src, [BUF_SIZE], [2]);

    declare_func(void, INTFLOAT *dst,
                 const INTFLOAT (*src)[2], int n);

    randomize((INTFLOAT *)src, BUF_SIZE * 2);
    randomize(dst0, BUF_SIZE);
    memcpy(dst1, dst0, BUF_SIZE * sizeof(INTFLOAT));
    call_ref(dst0, src, BUF_SIZE);
    call_new(dst1, src, BUF_SIZE);
    if (!float_near_abs_eps_array(dst0, dst1, EPS, BUF_SIZE))
        fail();
    bench_new(dst1, src, BUF_SIZE);
}

static void test_mul_pair_single(void)
{
    LOCAL_ALIGNED_16(INTFLOAT, dst0, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, dst1, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, src0, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, src1, [BUF_SIZE]);

    declare_func(void, INTFLOAT (*dst)[2],
                       INTFLOAT (*src0)[2], INTFLOAT *src1, int n);

    randomize((INTFLOAT *)src0, BUF_SIZE * 2);
    randomize(src1, BUF_SIZE);
    call_ref(dst0, src0, src1, BUF_SIZE);
    call_new(dst1, src0, src1, BUF_SIZE);
    if (!float_near_abs_eps_array((float *)dst0, (float *)dst1, EPS, BUF_SIZE * 2))
        fail();
    bench_new(dst1, src0, src1, BUF_SIZE);
}

static void test_hybrid_analysis(void)
{
    LOCAL_ALIGNED_16(INTFLOAT, dst0, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, dst1, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, in, [12], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, filter, [N], [8][2]);

    declare_func(void, INTFLOAT (*out)[2], INTFLOAT (*in)[2],
                 const INTFLOAT (*filter)[8][2],
                 int stride, int n);

    randomize((INTFLOAT *)in, 12 * 2);
    randomize((INTFLOAT *)filter, N * 8 * 2);

    randomize((INTFLOAT *)dst0, BUF_SIZE * 2);
    memcpy(dst1, dst0, BUF_SIZE * 2 * sizeof(INTFLOAT));

    call_ref(dst0, in, filter, STRIDE, N);
    call_new(dst1, in, filter, STRIDE, N);

    if (!float_near_abs_eps_array((float *)dst0, (float *)dst1, EPS, BUF_SIZE * 2))
        fail();
    bench_new(dst1, in, filter, STRIDE, N);
}

static void test_stereo_interpolate(PSDSPContext *psdsp)
{
    int i;
    LOCAL_ALIGNED_16(INTFLOAT, l,  [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, r,  [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, l0, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, r0, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, l1, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, r1, [BUF_SIZE], [2]);
    LOCAL_ALIGNED_16(INTFLOAT, h, [2], [4]);
    LOCAL_ALIGNED_16(INTFLOAT, h_step, [2], [4]);

    declare_func(void, INTFLOAT (*l)[2], INTFLOAT (*r)[2],
                       INTFLOAT h[2][4], INTFLOAT h_step[2][4], int len);

    randomize((INTFLOAT *)l, BUF_SIZE * 2);
    randomize((INTFLOAT *)r, BUF_SIZE * 2);

    for (i = 0; i < 2; i++) {
        if (check_func(psdsp->stereo_interpolate[i], "ps_stereo_interpolate%s", i ? "_ipdopd" : "")) {
            memcpy(l0, l, BUF_SIZE * 2 * sizeof(INTFLOAT));
            memcpy(l1, l, BUF_SIZE * 2 * sizeof(INTFLOAT));
            memcpy(r0, r, BUF_SIZE * 2 * sizeof(INTFLOAT));
            memcpy(r1, r, BUF_SIZE * 2 * sizeof(INTFLOAT));

            randomize((INTFLOAT *)h, 2 * 4);
            randomize((INTFLOAT *)h_step, 2 * 4);

            call_ref(l0, r0, h, h_step, BUF_SIZE);
            call_new(l1, r1, h, h_step, BUF_SIZE);
            if (!float_near_abs_eps_array((float *)l0, (float *)l1, EPS, BUF_SIZE * 2) ||
                !float_near_abs_eps_array((float *)r0, (float *)r1, EPS, BUF_SIZE * 2))
                fail();

            memcpy(l1, l, BUF_SIZE * 2 * sizeof(INTFLOAT));
            memcpy(r1, r, BUF_SIZE * 2 * sizeof(INTFLOAT));
            bench_new(l1, r1, h, h_step, BUF_SIZE);
        }
    }
}

void checkasm_check_aacpsdsp(void)
{
    PSDSPContext psdsp;

    ff_psdsp_init(&psdsp);

    if (check_func(psdsp.add_squares, "ps_add_squares"))
        test_add_squares();
    report("add_squares");

    if (check_func(psdsp.mul_pair_single, "ps_mul_pair_single"))
        test_mul_pair_single();
    report("mul_pair_single");

    if (check_func(psdsp.hybrid_analysis, "ps_hybrid_analysis"))
        test_hybrid_analysis();
    report("hybrid_analysis");

    test_stereo_interpolate(&psdsp);
    report("stereo_interpolate");
}
