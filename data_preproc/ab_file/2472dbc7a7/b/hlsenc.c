/*
 * Apple HTTP Live Streaming segmenter
 * Copyright (c) 2012, Luca Barbato
 * Copyright (c) 2017 Akamai Technologies, Inc.
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
#if CONFIG_HTTP_PROTOCOL
#include "http.h"
#endif
#include "hlsplaylist.h"
#include "internal.h"
#include "os_support.h"

typedef enum {
  HLS_START_SEQUENCE_AS_START_NUMBER = 0,
  HLS_START_SEQUENCE_AS_SECONDS_SINCE_EPOCH = 1,
  HLS_START_SEQUENCE_AS_FORMATTED_DATETIME = 2,  // YYYYMMDDhhmmss
} StartSequenceSourceType;

typedef enum {
    CODEC_ATTRIBUTE_WRITTEN = 0,
    CODEC_ATTRIBUTE_WILL_NOT_BE_WRITTEN,
} CodecAttributeStatus;

#define KEYSIZE 16
#define LINE_BUFFER_SIZE 1024
#define HLS_MICROSECOND_UNIT   1000000
#define POSTFIX_PATTERN "_%d"

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
    HLS_PERIODIC_REKEY = (1 << 12),
    HLS_INDEPENDENT_SEGMENTS = (1 << 13),
} HLSFlags;

typedef enum {
    SEGMENT_TYPE_MPEGTS,
    SEGMENT_TYPE_FMP4,
} SegmentType;

typedef struct VariantStream {
    unsigned number;
    int64_t sequence;
    AVOutputFormat *oformat;
    AVOutputFormat *vtt_oformat;
    AVIOContext *out;
    int packets_written;
    int init_range_length;

    AVFormatContext *avf;
    AVFormatContext *vtt_avf;

    int has_video;
    int has_subtitle;
    int new_start;
    double dpp;           // duration per packet
    int64_t start_pts;
    int64_t end_pts;
    double duration;      // last segment duration computed so far, in seconds
    int64_t start_pos;    // last segment starting position
    int64_t size;         // last segment size
    int nb_entries;
    int discontinuity_set;
    int discontinuity;

    HLSSegment *segments;
    HLSSegment *last_segment;
    HLSSegment *old_segments;

    char *basename;
    char *vtt_basename;
    char *vtt_m3u8_name;
    char *m3u8_name;

    double initial_prog_date_time;
    char current_segment_final_filename_fmt[1024]; // when renaming segments

    char *fmp4_init_filename;
    char *base_output_dirname;
    int fmp4_init_mode;

    AVStream **streams;
    char codec_attr[128];
    CodecAttributeStatus attr_status;
    unsigned int nb_streams;
    int m3u8_created; /* status of media play-list creation */
    char *agroup; /* audio group name */
    char *baseurl;
} VariantStream;

typedef struct HLSContext {
    const AVClass *class;  // Class for private options.
    int64_t start_sequence;
    uint32_t start_sequence_source_type;  // enum StartSequenceSourceType

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

    int use_localtime;      ///< flag to expand filename with localtime
    int use_localtime_mkdir;///< flag to mkdir dirname in timebased filename
    int allowcache;
    int64_t recording_time;
    int64_t max_seg_size; // every segment file max size

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
    int encrypt_started;

    char *key_info_file;
    char key_file[LINE_BUFFER_SIZE + 1];
    char key_uri[LINE_BUFFER_SIZE + 1];
    char key_string[KEYSIZE*2 + 1];
    char iv_string[KEYSIZE*2 + 1];
    AVDictionary *vtt_format_options;

    char *method;
    char *user_agent;

    VariantStream *var_streams;
    unsigned int nb_varstreams;

    int master_m3u8_created; /* status of master play-list creation */
    char *master_m3u8_url; /* URL of the master m3u8 file */
    int version; /* HLS version */
    char *var_stream_map; /* user specified variant stream map string */
    char *master_pl_name;
    unsigned int master_publish_rate;
    int http_persistent;
    AVIOContext *m3u8_out;
    AVIOContext *sub_m3u8_out;
} HLSContext;



















































































static void write_codec_attr(AVStream *st, VariantStream *vs) {
    int codec_strlen = strlen(vs->codec_attr);
    char attr[32];

    if (st->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)
        return;
    if (vs->attr_status == CODEC_ATTRIBUTE_WILL_NOT_BE_WRITTEN)
        return;

    if (st->codecpar->codec_id == AV_CODEC_ID_H264) {
        uint8_t *data = st->codecpar->extradata;
        if (data && (data[0] | data[1] | data[2]) == 0 && data[3] == 1 && (data[4] & 0x1F) == 7) {
            snprintf(attr, sizeof(attr),
                     "avc1.%02x%02x%02x", data[5], data[6], data[7]);
        } else {
            goto fail;
        }
    } else if (st->codecpar->codec_id == AV_CODEC_ID_MP2) {
        snprintf(attr, sizeof(attr), "mp4a.40.33");
    } else if (st->codecpar->codec_id == AV_CODEC_ID_MP3) {
        snprintf(attr, sizeof(attr), "mp4a.40.34");
    } else if (st->codecpar->codec_id == AV_CODEC_ID_AAC) {
        /* TODO : For HE-AAC, HE-AACv2, the last digit needs to be set to 5 and 29 respectively */
        snprintf(attr, sizeof(attr), "mp4a.40.2");
    } else if (st->codecpar->codec_id == AV_CODEC_ID_AC3) {
        snprintf(attr, sizeof(attr), "ac-3");
    } else if (st->codecpar->codec_id == AV_CODEC_ID_EAC3) {
        snprintf(attr, sizeof(attr), "ec-3");
    } else {
        goto fail;
    }
    // Don't write the same attribute multiple times
    if (!av_stristr(vs->codec_attr, attr)) {
        snprintf(vs->codec_attr + codec_strlen,
                 sizeof(vs->codec_attr) - codec_strlen,
                 "%s%s", codec_strlen ? "," : "", attr);
    }
    return;

fail:
    vs->codec_attr[0] = '\0';
    vs->attr_status = CODEC_ATTRIBUTE_WILL_NOT_BE_WRITTEN;
    return;
}




























































































































































































































































































































































































































































































































































































































/* Create a new segment and append it to the segment list */




















































































































































































































static int create_master_playlist(AVFormatContext *s,
                                  VariantStream * const input_vs)
{
    HLSContext *hls = s->priv_data;
    VariantStream *vs, *temp_vs;
    AVStream *vid_st, *aud_st;
    AVDictionary *options = NULL;
    unsigned int i, j;
    int m3u8_name_size, ret, bandwidth;
    char *m3u8_rel_name;

    input_vs->m3u8_created = 1;
    if (!hls->master_m3u8_created) {
        /* For the first time, wait until all the media playlists are created */
        for (i = 0; i < hls->nb_varstreams; i++)
            if (!hls->var_streams[i].m3u8_created)
                return 0;
    } else {
         /* Keep publishing the master playlist at the configured rate */
        if (&hls->var_streams[0] != input_vs || !hls->master_publish_rate ||
            input_vs->number % hls->master_publish_rate)
            return 0;
    }

    set_http_options(s, &options, hls);

    ret = hlsenc_io_open(s, &hls->m3u8_out, hls->master_m3u8_url, &options);
    av_dict_free(&options);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to open master play list file '%s'\n",
                hls->master_m3u8_url);
        goto fail;
    }

    ff_hls_write_playlist_version(hls->m3u8_out, hls->version);

    /* For audio only variant streams add #EXT-X-MEDIA tag with attributes*/
    for (i = 0; i < hls->nb_varstreams; i++) {
        vs = &(hls->var_streams[i]);

        if (vs->has_video || vs->has_subtitle || !vs->agroup)
            continue;

        m3u8_name_size = strlen(vs->m3u8_name) + 1;

        if (!m3u8_rel_name) {

            goto fail;
        }
        av_strlcpy(m3u8_rel_name, vs->m3u8_name, m3u8_name_size);
        ret = get_relative_url(hls->master_m3u8_url, vs->m3u8_name,
                               m3u8_rel_name, m3u8_name_size);
        if (ret < 0) {
            av_log(s, AV_LOG_ERROR, "Unable to find relative URL\n");
            goto fail;
        }

        ff_hls_write_audio_rendition(hls->m3u8_out, vs->agroup, m3u8_rel_name, 0, 1);

        av_freep(&m3u8_rel_name);
    }

    /* For variant streams with video add #EXT-X-STREAM-INF tag with attributes*/
    for (i = 0; i < hls->nb_varstreams; i++) {
        vs = &(hls->var_streams[i]);

        m3u8_name_size = strlen(vs->m3u8_name) + 1;

        if (!m3u8_rel_name) {

            goto fail;
        }
        av_strlcpy(m3u8_rel_name, vs->m3u8_name, m3u8_name_size);
        ret = get_relative_url(hls->master_m3u8_url, vs->m3u8_name,
                               m3u8_rel_name, m3u8_name_size);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Unable to find relative URL\n");
            goto fail;
        }

        vid_st = NULL;
        aud_st = NULL;
        for (j = 0; j < vs->nb_streams; j++) {
            if (vs->streams[j]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                vid_st = vs->streams[j];
            else if (vs->streams[j]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
                aud_st = vs->streams[j];
        }

        if (!vid_st && !aud_st) {
            av_log(NULL, AV_LOG_WARNING, "Media stream not found\n");
            continue;
        }

        /**
         * Traverse through the list of audio only rendition streams and find
         * the rendition which has highest bitrate in the same audio group
         */
        if (vs->agroup) {
            for (j = 0; j < hls->nb_varstreams; j++) {
                temp_vs = &(hls->var_streams[j]);
                if (!temp_vs->has_video && !temp_vs->has_subtitle &&
                    temp_vs->agroup &&
                    !av_strcasecmp(temp_vs->agroup, vs->agroup)) {
                    if (!aud_st)
                        aud_st = temp_vs->streams[0];
                    if (temp_vs->streams[0]->codecpar->bit_rate >
                            aud_st->codecpar->bit_rate)
                        aud_st = temp_vs->streams[0];
                }
            }
        }

        bandwidth = 0;
        if (vid_st)
            bandwidth += vid_st->codecpar->bit_rate;
        if (aud_st)
            bandwidth += aud_st->codecpar->bit_rate;
        bandwidth += bandwidth / 10;

        ff_hls_write_stream_info(vid_st, hls->m3u8_out, bandwidth, m3u8_rel_name,
                aud_st ? vs->agroup : NULL, vs->codec_attr);

        av_freep(&m3u8_rel_name);
    }
fail:
    if(ret >=0)
        hls->master_m3u8_created = 1;
    av_freep(&m3u8_rel_name);
    hlsenc_io_close(s, &hls->m3u8_out, hls->master_m3u8_url);
    return ret;
}










































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































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
    {"periodic_rekey", "reload keyinfo file periodically for re-keying", 0, AV_OPT_TYPE_CONST, {.i64 = HLS_PERIODIC_REKEY }, 0, UINT_MAX,   E, "flags"},
    {"independent_segments", "add EXT-X-INDEPENDENT-SEGMENTS, whenever applicable", 0, AV_OPT_TYPE_CONST, { .i64 = HLS_INDEPENDENT_SEGMENTS }, 0, UINT_MAX, E, "flags"},
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
    {"http_user_agent", "override User-Agent field in HTTP header", OFFSET(user_agent), AV_OPT_TYPE_STRING, {.str = NULL},  0, 0,    E},
    {"var_stream_map", "Variant stream map string", OFFSET(var_stream_map), AV_OPT_TYPE_STRING, {.str = NULL},  0, 0,    E},
    {"master_pl_name", "Create HLS master playlist with this name", OFFSET(master_pl_name), AV_OPT_TYPE_STRING, {.str = NULL},  0, 0,    E},
    {"master_pl_publish_rate", "Publish master play list every after this many segment intervals", OFFSET(master_publish_rate), AV_OPT_TYPE_INT, {.i64 = 0}, 0, UINT_MAX, E},
    {"http_persistent", "Use persistent HTTP connections", OFFSET(http_persistent), AV_OPT_TYPE_BOOL, {.i64 = 0 }, 0, 1, E },
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
    .init           = hls_init,
    .write_header   = hls_write_header,
    .write_packet   = hls_write_packet,
    .write_trailer  = hls_write_trailer,
    .priv_class     = &hls_class,
};
