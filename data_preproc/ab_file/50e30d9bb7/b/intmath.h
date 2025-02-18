/*
 * Copyright (c) 2015 James Almer
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

#ifndef AVUTIL_X86_INTMATH_H
#define AVUTIL_X86_INTMATH_H

#include <stdint.h>
#include <stdlib.h>
#if HAVE_FAST_CLZ
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__INTEL_COMPILER)
#include <immintrin.h>
#endif
#endif
#include "config.h"

#if HAVE_FAST_CLZ
#if (defined(__INTEL_COMPILER) && (__INTEL_COMPILER>=1216)) || defined(_MSC_VER)
#   if defined(__INTEL_COMPILER)
#       define ff_log2(x) (_bit_scan_reverse((x)|1))
#   else
#       define ff_log2 ff_log2_x86






#   endif
#   define ff_log2_16bit av_log2

#if defined(__INTEL_COMPILER) || (defined(_MSC_VER) && (_MSC_VER >= 1700) && \
                                  (defined(__BMI__) || !defined(__clang__)))
#   define ff_ctz(v) _tzcnt_u32(v)

#   if ARCH_X86_64
#       define ff_ctzll(v) _tzcnt_u64(v)
#   else
#       define ff_ctzll ff_ctzll_x86




#   endif
#endif /* _MSC_VER */

#endif /* __INTEL_COMPILER */

#endif /* HAVE_FAST_CLZ */

#if defined(__GNUC__)

/* Our generic version of av_popcount is faster than GCC's built-in on
 * CPUs that don't support the popcnt instruction.
 */
#if defined(__POPCNT__)
    #define av_popcount   __builtin_popcount
#if ARCH_X86_64
    #define av_popcount64 __builtin_popcountll
#endif

#endif /* __POPCNT__ */

#if defined(__BMI2__)

#if AV_GCC_VERSION_AT_LEAST(5,1)
#define av_mod_uintp2 __builtin_ia32_bzhi_si
#elif HAVE_INLINE_ASM
/* GCC releases before 5.1.0 have a broken bzhi builtin, so for those we
 * implement it using inline assembly
 */
#define av_mod_uintp2 av_mod_uintp2_bmi2










#endif /* AV_GCC_VERSION_AT_LEAST */

#endif /* __BMI2__ */

#if defined(__SSE2__) && !defined(__INTEL_COMPILER)

#define av_clipd av_clipd_sse2











#endif /* __SSE2__ */

#if defined(__SSE__) && !defined(__INTEL_COMPILER)

#define av_clipf av_clipf_sse











#endif /* __SSE__ */

#endif /* __GNUC__ */

#endif /* AVUTIL_X86_INTMATH_H */
