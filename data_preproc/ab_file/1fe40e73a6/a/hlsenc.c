/*
 * Apple HTTP Live Streaming segmenter
 * Copyright (c) 2012, Luca Barbato
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

#include "config.h"
#include <float.h>
#include <stdint.h>
#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#if CONFIG_GCRYPT
#include <gcrypt.h>
#elif CONFIG_OPENSSL
#include <openssl/rand.h>
#endif

#include "libavutil/avassert.h"
#include "libavutil/mathematics.h"
#include "libavutil/parseutils.h"
#include "libavutil/avstring.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/random_seed.h"
#include "libavutil/opt.h"
#include "libavutil/log.h"
#include "libavutil/time_internal.h"

#include "avformat.h"
#include "avio_internal.h"
#include "internal.h"
#include "os_support.h"

typedef enum {
  HLS_START_SEQUENCE_AS_START_NUMBER = 0,
  HLS_START_SEQUENCE_AS_SECONDS_SINCE_EPOCH = 1,
  HLS_START_SEQUENCE_AS_FORMATTED_DATETIME = 2,  // YYYYMMDDhhmmss
} StartSequenceSourceType;

#define KEYSIZE 16
#define LINE_BUFFER_SIZE 1024
#define HLS_MICROSECOND_UNIT   1000000

typedef struct HLSSegment {
    char filename[1024];
    char sub_filename[1024];
    double duration; /* in seconds */
    int discont;
    int64_t pos;
    int64_t size;

    char key_uri[LINE_BUFFER_SIZE + 1];
    char iv_string[KEYSIZE*2 + 1];

    struct HLSSegment *next;
} HLSSegment;

typedef enum HLSFlags {
    // Generate a single media file and use byte ranges in the playlist.
    HLS_SINGLE_FILE = (1 << 0),
    HLS_DELETE_SEGMENTS = (1 << 1),
    HLS_ROUND_DURATIONS = (1 << 2),
    HLS_DISCONT_START = (1 << 3),
    HLS_OMIT_ENDLIST = (1 << 4),
    HLS_SPLIT_BY_TIME = (1 << 5),
    HLS_APPEND_LIST = (1 << 6),
    HLS_PROGRAM_DATE_TIME = (1 << 7),
    HLS_SECOND_LEVEL_SEGMENT_INDEX = (1 << 8), // include segment index in segment filenames when use_localtime  e.g.: %%03d
    HLS_SECOND_LEVEL_SEGMENT_DURATION = (1 << 9), // include segment duration (microsec) in segment filenames when use_localtime  e.g.: %%09t
    HLS_SECOND_LEVEL_SEGMENT_SIZE = (1 << 10), // include segment size (bytes) in segment filenames when use_localtime  e.g.: %%014s
    HLS_TEMP_FILE = (1 << 11),
} HLSFlags;

typedef enum {
    SEGMENT_TYPE_MPEGTS,
    SEGMENT_TYPE_FMP4,
} SegmentType;

typedef enum {
    PLAYLIST_TYPE_NONE,
    PLAYLIST_TYPE_EVENT,
    PLAYLIST_TYPE_VOD,
    PLAYLIST_TYPE_NB,
} PlaylistType;

typedef struct HLSContext {
    const AVClass *class;  // Class for private options.
    unsigned number;
    int64_t sequence;
    int64_t start_sequence;
    uint32_t start_sequence_source_type;  // enum StartSequenceSourceType
    AVOutputFormat *oformat;
    AVOutputFormat *vtt_oformat;

    AVFormatContext *avf;
    AVFormatContext *vtt_avf;

    float time;            // Set by a private option.
    float init_time;       // Set by a private option.
    int max_nb_segments;   // Set by a private option.
#if FF_API_HLS_WRAP
    int  wrap;             // Set by a private option.
#endif
    uint32_t flags;        // enum HLSFlags
    uint32_t pl_type;      // enum PlaylistType
    char *segment_filename;
    char *fmp4_init_filename;
    int segment_type;
    int fmp4_init_mode;

    int use_localtime;      ///< flag to expand filename with localtime
    int use_localtime_mkdir;///< flag to mkdir dirname in timebased filename
    int allowcache;
    int64_t recording_time;
    int has_video;
    int has_subtitle;
    int new_start;
    double dpp;           // duration per packet
    int64_t start_pts;
    int64_t end_pts;
    double duration;      // last segment duration computed so far, in seconds
    int64_t start_pos;    // last segment starting position
    int64_t size;         // last segment size
    int64_t max_seg_size; // every segment file max size
    int nb_entries;
    int discontinuity_set;
    int discontinuity;

    HLSSegment *segments;
    HLSSegment *last_segment;
    HLSSegment *old_segments;

    char *basename;
    char *vtt_basename;
    char *vtt_m3u8_name;
    char *baseurl;
    char *format_options_str;
    char *vtt_format_options_str;
    char *subtitle_filename;
    AVDictionary *format_options;

    int encrypt;
    char *key;
    char *key_url;
    char *iv;
    char *key_basename;

    char *key_info_file;
    char key_file[LINE_BUFFER_SIZE + 1];
    char key_uri[LINE_BUFFER_SIZE + 1];
    char key_string[KEYSIZE*2 + 1];
    char iv_string[KEYSIZE*2 + 1];
    AVDictionary *vtt_format_options;

    char *method;

    double initial_prog_date_time;
    char current_segment_final_filename_fmt[1024]; // when renaming segments
} HLSContext;































































































































































































































































































































































static int hls_mux_init(AVFormatContext *s)
{
    AVDictionary *options = NULL;
    HLSContext *hls = s->priv_data;
    AVFormatContext *oc;
    AVFormatContext *vtt_oc = NULL;
    int i, ret;

    ret = avformat_alloc_output_context2(&hls->avf, hls->oformat, NULL, NULL);
    if (ret < 0)
        return ret;
    oc = hls->avf;

    oc->filename[0]        = '\0';
    oc->oformat            = hls->oformat;
    oc->interrupt_callback = s->interrupt_callback;
    oc->max_delay          = s->max_delay;
    oc->opaque             = s->opaque;
    oc->io_open            = s->io_open;
    oc->io_close           = s->io_close;
    av_dict_copy(&oc->metadata, s->metadata, 0);

    if(hls->vtt_oformat) {
        ret = avformat_alloc_output_context2(&hls->vtt_avf, hls->vtt_oformat, NULL, NULL);
        if (ret < 0)
            return ret;
        vtt_oc          = hls->vtt_avf;
        vtt_oc->oformat = hls->vtt_oformat;
        av_dict_copy(&vtt_oc->metadata, s->metadata, 0);
    }

    for (i = 0; i < s->nb_streams; i++) {
        AVStream *st;
        AVFormatContext *loc;
        if (s->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)
            loc = vtt_oc;
        else
            loc = oc;

        if (!(st = avformat_new_stream(loc, NULL)))
            return AVERROR(ENOMEM);
        avcodec_parameters_copy(st->codecpar, s->streams[i]->codecpar);








        st->sample_aspect_ratio = s->streams[i]->sample_aspect_ratio;
        st->time_base = s->streams[i]->time_base;
        av_dict_copy(&st->metadata, s->streams[i]->metadata, 0);
    }
    hls->start_pos = 0;
    hls->new_start = 1;
    hls->fmp4_init_mode = 0;

    if (hls->segment_type == SEGMENT_TYPE_FMP4) {
        hls->fmp4_init_mode = 1;
        if ((ret = s->io_open(s, &oc->pb, hls->fmp4_init_filename, AVIO_FLAG_WRITE, NULL)) < 0) {
            av_log(s, AV_LOG_ERROR, "Failed to open segment '%s'\n", hls->fmp4_init_filename);
            return ret;
        }

        if (hls->format_options_str) {
            ret = av_dict_parse_string(&hls->format_options, hls->format_options_str, "=", ":", 0);
            if (ret < 0) {
                av_log(s, AV_LOG_ERROR, "Could not parse format options list '%s'\n",
                       hls->format_options_str);
                return ret;
            }
        }

        av_dict_copy(&options, hls->format_options, 0);
        av_dict_set(&options, "fflags", "-autobsf", 0);
        av_dict_set(&options, "movflags", "frag_custom+dash+delay_moov", 0);
        ret = avformat_init_output(oc, &options);
        if (av_dict_count(options)) {
            av_log(s, AV_LOG_ERROR, "Some of the provided format options in '%s' are not recognized\n", hls->format_options_str);
            av_dict_free(&options);
            return AVERROR(EINVAL);
        }
        av_dict_free(&options);
    }
    return 0;
}
































































































































































/* Create a new segment and append it to the segment list */





























































































































































































































































































































































































































































































































































































































































































































































































































































































































































#define OFFSET(x) offsetof(HLSContext, x)
#define E AV_OPT_FLAG_ENCODING_PARAM
static const AVOption options[] = {
    {"start_number",  "set first number in the sequence",        OFFSET(start_sequence),AV_OPT_TYPE_INT64,  {.i64 = 0},     0, INT64_MAX, E},
    {"hls_time",      "set segment length in seconds",           OFFSET(time),    AV_OPT_TYPE_FLOAT,  {.dbl = 2},     0, FLT_MAX, E},
    {"hls_init_time", "set segment length in seconds at init list",           OFFSET(init_time),    AV_OPT_TYPE_FLOAT,  {.dbl = 0},     0, FLT_MAX, E},
    {"hls_list_size", "set maximum number of playlist entries",  OFFSET(max_nb_segments),    AV_OPT_TYPE_INT,    {.i64 = 5},     0, INT_MAX, E},
    {"hls_ts_options","set hls mpegts list of options for the container format used for hls", OFFSET(format_options_str), AV_OPT_TYPE_STRING, {.str = NULL},  0, 0,    E},
    {"hls_vtt_options","set hls vtt list of options for the container format used for hls", OFFSET(vtt_format_options_str), AV_OPT_TYPE_STRING, {.str = NULL},  0, 0,    E},
#if FF_API_HLS_WRAP
    {"hls_wrap",      "set number after which the index wraps (will be deprecated)",  OFFSET(wrap),    AV_OPT_TYPE_INT,    {.i64 = 0},     0, INT_MAX, E},
#endif
    {"hls_allow_cache", "explicitly set whether the client MAY (1) or MUST NOT (0) cache media segments", OFFSET(allowcache), AV_OPT_TYPE_INT, {.i64 = -1}, INT_MIN, INT_MAX, E},
    {"hls_base_url",  "url to prepend to each playlist entry",   OFFSET(baseurl), AV_OPT_TYPE_STRING, {.str = NULL},  0, 0,       E},
    {"hls_segment_filename", "filename template for segment files", OFFSET(segment_filename),   AV_OPT_TYPE_STRING, {.str = NULL},            0,       0,         E},
    {"hls_segment_size", "maximum size per segment file, (in bytes)",  OFFSET(max_seg_size),    AV_OPT_TYPE_INT,    {.i64 = 0},               0,       INT_MAX,   E},
    {"hls_key_info_file",    "file with key URI and key file path", OFFSET(key_info_file),      AV_OPT_TYPE_STRING, {.str = NULL},            0,       0,         E},
    {"hls_enc",    "enable AES128 encryption support", OFFSET(encrypt),      AV_OPT_TYPE_BOOL, {.i64 = 0},            0,       1,         E},
    {"hls_enc_key",    "hex-coded 16 byte key to encrypt the segments", OFFSET(key),      AV_OPT_TYPE_STRING, .flags = E},
    {"hls_enc_key_url",    "url to access the key to decrypt the segments", OFFSET(key_url),      AV_OPT_TYPE_STRING, {.str = NULL},            0,       0,         E},
    {"hls_enc_iv",    "hex-coded 16 byte initialization vector", OFFSET(iv),      AV_OPT_TYPE_STRING, .flags = E},
    {"hls_subtitle_path",     "set path of hls subtitles", OFFSET(subtitle_filename), AV_OPT_TYPE_STRING, {.str = NULL},  0, 0,    E},
    {"hls_segment_type",     "set hls segment files type", OFFSET(segment_type), AV_OPT_TYPE_INT, {.i64 = SEGMENT_TYPE_MPEGTS }, 0, SEGMENT_TYPE_FMP4, E, "segment_type"},
    {"mpegts",   "make segment file to mpegts files in m3u8", 0, AV_OPT_TYPE_CONST, {.i64 = SEGMENT_TYPE_MPEGTS }, 0, UINT_MAX,   E, "segment_type"},
    {"fmp4",   "make segment file to fragment mp4 files in m3u8", 0, AV_OPT_TYPE_CONST, {.i64 = SEGMENT_TYPE_FMP4 }, 0, UINT_MAX,   E, "segment_type"},
    {"hls_fmp4_init_filename", "set fragment mp4 file init filename", OFFSET(fmp4_init_filename),   AV_OPT_TYPE_STRING, {.str = "init.mp4"},            0,       0,         E},
    {"hls_flags",     "set flags affecting HLS playlist and media file generation", OFFSET(flags), AV_OPT_TYPE_FLAGS, {.i64 = 0 }, 0, UINT_MAX, E, "flags"},
    {"single_file",   "generate a single media file indexed with byte ranges", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_SINGLE_FILE }, 0, UINT_MAX,   E, "flags"},
    {"temp_file", "write segment to temporary file and rename when complete", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_TEMP_FILE }, 0, UINT_MAX,   E, "flags"},
    {"delete_segments", "delete segment files that are no longer part of the playlist", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_DELETE_SEGMENTS }, 0, UINT_MAX,   E, "flags"},
    {"round_durations", "round durations in m3u8 to whole numbers", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_ROUND_DURATIONS }, 0, UINT_MAX,   E, "flags"},
    {"discont_start", "start the playlist with a discontinuity tag", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_DISCONT_START }, 0, UINT_MAX,   E, "flags"},
    {"omit_endlist", "Do not append an endlist when ending stream", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_OMIT_ENDLIST }, 0, UINT_MAX,   E, "flags"},
    {"split_by_time", "split the hls segment by time which user set by hls_time", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_SPLIT_BY_TIME }, 0, UINT_MAX,   E, "flags"},
    {"append_list", "append the new segments into old hls segment list", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_APPEND_LIST }, 0, UINT_MAX,   E, "flags"},
    {"program_date_time", "add EXT-X-PROGRAM-DATE-TIME", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_PROGRAM_DATE_TIME }, 0, UINT_MAX,   E, "flags"},
    {"second_level_segment_index", "include segment index in segment filenames when use_localtime", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_SECOND_LEVEL_SEGMENT_INDEX }, 0, UINT_MAX,   E, "flags"},
    {"second_level_segment_duration", "include segment duration in segment filenames when use_localtime", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_SECOND_LEVEL_SEGMENT_DURATION }, 0, UINT_MAX,   E, "flags"},
    {"second_level_segment_size", "include segment size in segment filenames when use_localtime", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_SECOND_LEVEL_SEGMENT_SIZE }, 0, UINT_MAX,   E, "flags"},
    {"use_localtime", "set filename expansion with strftime at segment creation", OFFSET(use_localtime), AV_OPT_TYPE_BOOL, {.i64 = 0 }, 0, 1, E },
    {"use_localtime_mkdir", "create last directory component in strftime-generated filename", OFFSET(use_localtime_mkdir), AV_OPT_TYPE_BOOL, {.i64 = 0 }, 0, 1, E },
    {"hls_playlist_type", "set the HLS playlist type", OFFSET(pl_type), AV_OPT_TYPE_INT, {.i64 = PLAYLIST_TYPE_NONE }, 0, PLAYLIST_TYPE_NB-1, E, "pl_type" },
    {"event", "EVENT playlist", 0, AV_OPT_TYPE_CONST, {.i64 = PLAYLIST_TYPE_EVENT }, INT_MIN, INT_MAX, E, "pl_type" },
    {"vod", "VOD playlist", 0, AV_OPT_TYPE_CONST, {.i64 = PLAYLIST_TYPE_VOD }, INT_MIN, INT_MAX, E, "pl_type" },
    {"method", "set the HTTP method(default: PUT)", OFFSET(method), AV_OPT_TYPE_STRING, {.str = NULL},  0, 0,    E},
    {"hls_start_number_source", "set source of first number in sequence", OFFSET(start_sequence_source_type), AV_OPT_TYPE_INT, {.i64 = HLS_START_SEQUENCE_AS_START_NUMBER }, 0, HLS_START_SEQUENCE_AS_FORMATTED_DATETIME, E, "start_sequence_source_type" },
    {"generic", "start_number value (default)", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_START_SEQUENCE_AS_START_NUMBER }, INT_MIN, INT_MAX, E, "start_sequence_source_type" },
    {"epoch", "seconds since epoch", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_START_SEQUENCE_AS_SECONDS_SINCE_EPOCH }, INT_MIN, INT_MAX, E, "start_sequence_source_type" },
    {"datetime", "current datetime as YYYYMMDDhhmmss", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_START_SEQUENCE_AS_FORMATTED_DATETIME }, INT_MIN, INT_MAX, E, "start_sequence_source_type" },
    { NULL },
};

static const AVClass hls_class = {
    .class_name = "hls muxer",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};


AVOutputFormat ff_hls_muxer = {
    .name           = "hls",
    .long_name      = NULL_IF_CONFIG_SMALL("Apple HTTP Live Streaming"),
    .extensions     = "m3u8",
    .priv_data_size = sizeof(HLSContext),
    .audio_codec    = AV_CODEC_ID_AAC,
    .video_codec    = AV_CODEC_ID_H264,
    .subtitle_codec = AV_CODEC_ID_WEBVTT,
    .flags          = AVFMT_NOFILE | AVFMT_GLOBALHEADER | AVFMT_ALLOW_FLUSH,
    .write_header   = hls_write_header,
    .write_packet   = hls_write_packet,
    .write_trailer  = hls_write_trailer,
    .priv_class     = &hls_class,
};
