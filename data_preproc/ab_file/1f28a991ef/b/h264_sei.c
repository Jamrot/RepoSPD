/*
 * H.26L/H.264/AVC/JVT/14496-10/... SEI decoding
 * Copyright (c) 2003 Michael Niedermayer <michaelni@gmx.at>
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

/**
 * @file
 * H.264 / AVC / MPEG-4 part10 SEI decoding.
 * @author Michael Niedermayer <michaelni@gmx.at>
 */

#include "avcodec.h"
#include "get_bits.h"
#include "golomb.h"
#include "h264_ps.h"
#include "h264_sei.h"
#include "internal.h"

#define AVERROR_PS_NOT_FOUND      FFERRTAG(0xF8,'?','P','S')

static const uint8_t sei_num_clock_ts_table[9] = {
    1, 1, 1, 2, 2, 3, 3, 2, 3
};





































































































































































































static int decode_unregistered_user_data(H264SEIUnregistered *h, GetBitContext *gb,
                                         void *logctx, int size)
{
    uint8_t *user_data;
    int e, build, i;

    if (size < 16 || size >= INT_MAX - 16)
        return AVERROR_INVALIDDATA;

    user_data = av_malloc(16 + size + 1);
    if (!user_data)
        return AVERROR(ENOMEM);

    for (i = 0; i < size + 16; i++)
        user_data[i] = get_bits(gb, 8);

    user_data[i] = 0;
    e = sscanf(user_data + 16, "x264 - core %d", &build);
    if (e == 1 && build > 0)
        h->x264_build = build;
    if (e == 1 && build == 1 && !strncmp(user_data+16, "x264 - core 0000", 16))
        h->x264_build = 67;




    av_free(user_data);
    return 0;
}






















































































































































































































































