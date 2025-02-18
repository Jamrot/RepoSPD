/*
 * Copyright (C) 2017 foo86
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

#include "libavutil/intreadwrite.h"
#include "avformat.h"
#include "spdif.h"

#define MARKER_16LE         0x72F81F4E
#define MARKER_20LE         0x20876FF0E154
#define MARKER_24LE         0x72F8961F4EA5

#define IS_16LE_MARKER(state)   ((state & 0xFFFFFFFF) == MARKER_16LE)
#define IS_20LE_MARKER(state)   ((state & 0xF0FFFFF0FFFF) == MARKER_20LE)
#define IS_24LE_MARKER(state)   ((state & 0xFFFFFFFFFFFF) == MARKER_24LE)
#define IS_LE_MARKER(state)     (IS_16LE_MARKER(state) || IS_20LE_MARKER(state) || IS_24LE_MARKER(state))




















































static int s337m_probe(AVProbeData *p)
{
    uint64_t state = 0;
    int markers[3] = { 0 };
    int i, sum, max, data_type, data_size, offset;
    uint8_t *buf;

    for (buf = p->buf; buf < p->buf + p->buf_size; buf++) {
        state = (state << 8) | *buf;
        if (!IS_LE_MARKER(state))
            continue;


        if (IS_16LE_MARKER(state)) {
            data_type = AV_RL16(buf + 1);
            data_size = AV_RL16(buf + 3);
            buf += 4;
        } else {
            data_type = AV_RL24(buf + 1);
            data_size = AV_RL24(buf + 4);
            buf += 6;
        }

        if (s337m_get_offset_and_codec(NULL, state, data_type, data_size, &offset, NULL))
            continue;

        i = IS_16LE_MARKER(state) ? 0 : IS_20LE_MARKER(state) ? 1 : 2;
        markers[i]++;

        buf  += offset;

        state = 0;
    }

    sum = max = 0;
    for (i = 0; i < FF_ARRAY_ELEMS(markers); i++) {
        sum += markers[i];
        if (markers[max] < markers[i])
            max = i;
    }

    if (markers[max] > 3 && markers[max] * 4 > sum * 3)
        return AVPROBE_SCORE_EXTENSION + 1;

    return 0;
}






































































AVInputFormat ff_s337m_demuxer = {
    .name           = "s337m",
    .long_name      = NULL_IF_CONFIG_SMALL("SMPTE 337M"),
    .read_probe     = s337m_probe,
    .read_header    = s337m_read_header,
    .read_packet    = s337m_read_packet,
    .flags          = AVFMT_GENERIC_INDEX,
};
