/*
 * Assembly testing and benchmarking tool
 * Copyright (c) 2015 Henrik Gramner
 * Copyright (c) 2008 Loren Merritt
 *
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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "checkasm.h"
#include "libavutil/common.h"
#include "libavutil/cpu.h"
#include "libavutil/intfloat.h"
#include "libavutil/random_seed.h"

#if HAVE_IO_H
#include <io.h>
#endif

#if HAVE_SETCONSOLETEXTATTRIBUTE
#include <windows.h>
#define COLOR_RED    FOREGROUND_RED
#define COLOR_GREEN  FOREGROUND_GREEN
#define COLOR_YELLOW (FOREGROUND_RED|FOREGROUND_GREEN)
#else
#define COLOR_RED    1
#define COLOR_GREEN  2
#define COLOR_YELLOW 3
#endif

#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#if !HAVE_ISATTY
#define isatty(fd) 1
#endif

#if ARCH_ARM && HAVE_ARMV5TE_EXTERNAL
#include "libavutil/arm/cpu.h"

void (*checkasm_checked_call)(void *func, int dummy, ...) = checkasm_checked_call_novfp;
#endif

/* List of tests to invoke */
static const struct {
    const char *name;
    void (*func)(void);
} tests[] = {
#if CONFIG_AVCODEC



    #if CONFIG_ALAC_DECODER
        { "alacdsp", checkasm_check_alacdsp },
    #endif
    #if CONFIG_AUDIODSP
        { "audiodsp", checkasm_check_audiodsp },
    #endif
    #if CONFIG_BLOCKDSP
        { "blockdsp", checkasm_check_blockdsp },
    #endif
    #if CONFIG_BSWAPDSP
        { "bswapdsp", checkasm_check_bswapdsp },
    #endif
    #if CONFIG_DCA_DECODER
        { "synth_filter", checkasm_check_synth_filter },
    #endif
    #if CONFIG_FLACDSP
        { "flacdsp", checkasm_check_flacdsp },
    #endif
    #if CONFIG_FMTCONVERT
        { "fmtconvert", checkasm_check_fmtconvert },
    #endif
    #if CONFIG_H264DSP
        { "h264dsp", checkasm_check_h264dsp },
    #endif
    #if CONFIG_H264PRED
        { "h264pred", checkasm_check_h264pred },
    #endif
    #if CONFIG_H264QPEL
        { "h264qpel", checkasm_check_h264qpel },
    #endif
    #if CONFIG_HEVC_DECODER
        { "hevc_add_res", checkasm_check_hevc_add_res },
        { "hevc_idct", checkasm_check_hevc_idct },
    #endif
    #if CONFIG_JPEG2000_DECODER
        { "jpeg2000dsp", checkasm_check_jpeg2000dsp },
    #endif
    #if CONFIG_HUFFYUVDSP
        { "llviddsp", checkasm_check_llviddsp },
    #endif
    #if CONFIG_PIXBLOCKDSP
        { "pixblockdsp", checkasm_check_pixblockdsp },
    #endif
    #if CONFIG_V210_ENCODER
        { "v210enc", checkasm_check_v210enc },
    #endif
    #if CONFIG_VP8DSP
        { "vp8dsp", checkasm_check_vp8dsp },
    #endif
    #if CONFIG_VP9_DECODER
        { "vp9dsp", checkasm_check_vp9dsp },
    #endif
    #if CONFIG_VIDEODSP
        { "videodsp", checkasm_check_videodsp },
    #endif
#endif
#if CONFIG_AVFILTER
    #if CONFIG_BLEND_FILTER
        { "vf_blend", checkasm_check_blend },
    #endif
    #if CONFIG_COLORSPACE_FILTER
        { "vf_colorspace", checkasm_check_colorspace },
    #endif
#endif
#if CONFIG_AVUTIL
        { "fixed_dsp", checkasm_check_fixed_dsp },
        { "float_dsp", checkasm_check_float_dsp },
#endif
    { NULL }
};

/* List of cpu flags to check */
static const struct {
    const char *name;
    const char *suffix;
    int flag;
} cpus[] = {
#if   ARCH_AARCH64
    { "ARMV8",    "armv8",    AV_CPU_FLAG_ARMV8 },
    { "NEON",     "neon",     AV_CPU_FLAG_NEON },
#elif ARCH_ARM
    { "ARMV5TE",  "armv5te",  AV_CPU_FLAG_ARMV5TE },
    { "ARMV6",    "armv6",    AV_CPU_FLAG_ARMV6 },
    { "ARMV6T2",  "armv6t2",  AV_CPU_FLAG_ARMV6T2 },
    { "VFP",      "vfp",      AV_CPU_FLAG_VFP },
    { "VFP_VM",   "vfp_vm",   AV_CPU_FLAG_VFP_VM },
    { "VFPV3",    "vfp3",     AV_CPU_FLAG_VFPV3 },
    { "NEON",     "neon",     AV_CPU_FLAG_NEON },
#elif ARCH_PPC
    { "ALTIVEC",  "altivec",  AV_CPU_FLAG_ALTIVEC },
    { "VSX",      "vsx",      AV_CPU_FLAG_VSX },
    { "POWER8",   "power8",   AV_CPU_FLAG_POWER8 },
#elif ARCH_X86
    { "MMX",      "mmx",      AV_CPU_FLAG_MMX|AV_CPU_FLAG_CMOV },
    { "MMXEXT",   "mmxext",   AV_CPU_FLAG_MMXEXT },
    { "3DNOW",    "3dnow",    AV_CPU_FLAG_3DNOW },
    { "3DNOWEXT", "3dnowext", AV_CPU_FLAG_3DNOWEXT },
    { "SSE",      "sse",      AV_CPU_FLAG_SSE },
    { "SSE2",     "sse2",     AV_CPU_FLAG_SSE2|AV_CPU_FLAG_SSE2SLOW },
    { "SSE3",     "sse3",     AV_CPU_FLAG_SSE3|AV_CPU_FLAG_SSE3SLOW },
    { "SSSE3",    "ssse3",    AV_CPU_FLAG_SSSE3|AV_CPU_FLAG_ATOM },
    { "SSE4.1",   "sse4",     AV_CPU_FLAG_SSE4 },
    { "SSE4.2",   "sse42",    AV_CPU_FLAG_SSE42 },
    { "AES-NI",   "aesni",    AV_CPU_FLAG_AESNI },
    { "AVX",      "avx",      AV_CPU_FLAG_AVX },
    { "XOP",      "xop",      AV_CPU_FLAG_XOP },
    { "FMA3",     "fma3",     AV_CPU_FLAG_FMA3 },
    { "FMA4",     "fma4",     AV_CPU_FLAG_FMA4 },
    { "AVX2",     "avx2",     AV_CPU_FLAG_AVX2 },
#endif
    { NULL }
};

typedef struct CheckasmFuncVersion {
    struct CheckasmFuncVersion *next;
    void *func;
    int ok;
    int cpu;
    int iterations;
    uint64_t cycles;
} CheckasmFuncVersion;

/* Binary search tree node */
typedef struct CheckasmFunc {
    struct CheckasmFunc *child[2];
    CheckasmFuncVersion versions;
    uint8_t color; /* 0 = red, 1 = black */
    char name[1];
} CheckasmFunc;

/* Internal state */
static struct {
    CheckasmFunc *funcs;
    CheckasmFunc *current_func;
    CheckasmFuncVersion *current_func_ver;
    const char *current_test_name;
    const char *bench_pattern;
    int bench_pattern_len;
    int num_checked;
    int num_failed;
    int nop_time;
    int cpu_flag;
    const char *cpu_flag_name;
    const char *test_name;
} state;

/* PRNG state */
AVLFG checkasm_lfg;

/* float compare support code */


























































































/* Print colored text to stderr if the terminal supports it */










































/* Deallocate a tree */
















/* Allocate a zero-initialized block, clean up and exit on failure */











/* Get the suffix of the specified cpu flag */











#ifdef AV_READ_TIME





/* Measure the overhead of the timing code (in decicycles) */

















/* Print benchmark results */



















#endif

/* ASCIIbetical sort except preserving natural order for numbers */














/* Perform a tree rotation in the specified direction and return the new root */










#define is_red(f) ((f) && !(f)->color)

/* Balance a left-leaning red-black tree at the specified node */















/* Get a node with the specified name, creating it if it doesn't exist */
























/* Perform tests and benchmarks for the specified cpu flag if supported by the host */






















/* Print the name of the current CPU flag, but only do it once */





































































/* Decide whether or not the specified function needs to be tested and
 * allocate/initialize data structures if needed. Returns a pointer to a
 * reference function if the function should be tested, otherwise NULL */














































/* Decide whether or not the current function needs to be benchmarked */






/* Indicate that the current test has failed */

















/* Update benchmark results of the current function */






/* Print the outcome of all tests performed since the last time this function was called */




































