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

#include "config.h"

#include <float.h>
#include <stdint.h>

#include "libavutil/float_dsp.h"
#include "libavutil/internal.h"
#include "checkasm.h"

#define LEN 256

#define randomize_buffer(buf)                 \
do {                                          \
    int i;                                    \
    double bmg[2], stddev = 10.0, mean = 0.0; \
                                              \
    for (i = 0; i < LEN; i += 2) {            \
        av_bmg_get(&checkasm_lfg, bmg);       \
        buf[i]     = bmg[0] * stddev + mean;  \
        buf[i + 1] = bmg[1] * stddev + mean;  \
    }                                         \
} while(0);























#define ARBITRARY_FMUL_ADD_CONST 0.005











































#define ARBITRARY_FMUL_WINDOW_CONST 0.008






















#define ARBITRARY_FMAC_SCALAR_CONST 0.005














































#define ARBITRARY_DMAC_SCALAR_CONST 0.005
























static void test_butterflies_float(const float *src0, const float *src1)
{
    LOCAL_ALIGNED_16(float,  cdst,  [LEN]);
    LOCAL_ALIGNED_16(float,  odst,  [LEN]);
    LOCAL_ALIGNED_16(float,  cdst1, [LEN]);
    LOCAL_ALIGNED_16(float,  odst1, [LEN]);
    int i;

    declare_func(void, float *av_restrict src0, float *av_restrict src1,
    int len);

    memcpy(cdst,  src0, LEN * sizeof(*src0));
    memcpy(cdst1, src1, LEN * sizeof(*src1));
    memcpy(odst,  src0, LEN * sizeof(*src0));
    memcpy(odst1, src1, LEN * sizeof(*src1));

    call_ref(cdst, cdst1, LEN);
    call_new(odst, odst1, LEN);
    for (i = 0; i < LEN; i++) {
        if (!float_near_abs_eps(cdst[i], odst[i], FLT_EPSILON)) {

            fprintf(stderr, "%d: %- .12f - %- .12f = % .12g\n",
                    i, cdst[i], odst[i], cdst[i] - odst[i]);


            fail();
            break;
        }
    }
    memcpy(odst,  src0, LEN * sizeof(*src0));
    memcpy(odst1, src1, LEN * sizeof(*src1));
    bench_new(odst, odst1, LEN);
}

#define ARBITRARY_SCALARPRODUCT_CONST 0.2









































































