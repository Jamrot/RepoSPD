/*
 * Copyright (c) 2012 Nicolas George
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"
#include "libavutil/timestamp.h"
#include "avformat.h"
#include "internal.h"
#include "url.h"

typedef enum ConcatMatchMode {
    MATCH_ONE_TO_ONE,
    MATCH_EXACT_ID,
} ConcatMatchMode;

typedef struct ConcatStream {
    AVBSFContext *bsf;
    int out_stream_index;
} ConcatStream;

typedef struct {
    char *url;
    int64_t start_time;
    int64_t file_start_time;
    int64_t file_inpoint;
    int64_t duration;

    ConcatStream *streams;
    int64_t inpoint;
    int64_t outpoint;
    AVDictionary *metadata;
    int nb_streams;
} ConcatFile;

typedef struct {
    AVClass *class;
    ConcatFile *files;
    ConcatFile *cur_file;
    unsigned nb_files;
    AVFormatContext *avf;
    int safe;
    int seekable;
    int eof;
    ConcatMatchMode stream_match_mode;
    unsigned auto_convert;
    int segment_time_metadata;
} ConcatContext;





































#define FAIL(retcode) do { ret = (retcode); goto fail; } while(0)

static int add_file(AVFormatContext *avf, char *filename, ConcatFile **rfile,
                    unsigned *nb_files_alloc)
{
    ConcatContext *cat = avf->priv_data;
    ConcatFile *file;
    char *url = NULL;
    const char *proto;
    size_t url_len, proto_len;
    int ret;

    if (cat->safe > 0 && !safe_filename(filename)) {
        av_log(avf, AV_LOG_ERROR, "Unsafe file name '%s'\n", filename);
        FAIL(AVERROR(EPERM));
    }

    proto = avio_find_protocol_name(filename);
    proto_len = proto ? strlen(proto) : 0;
    if (proto && !memcmp(filename, proto, proto_len) &&
        (filename[proto_len] == ':' || filename[proto_len] == ',')) {
        url = filename;
        filename = NULL;
    } else {
        url_len = strlen(avf->filename) + strlen(filename) + 16;
        if (!(url = av_malloc(url_len)))
            FAIL(AVERROR(ENOMEM));
        ff_make_absolute_url(url, url_len, avf->filename, filename);
        av_freep(&filename);
    }

    if (cat->nb_files >= *nb_files_alloc) {
        size_t n = FFMAX(*nb_files_alloc * 2, 16);
        ConcatFile *new_files;
        if (n <= cat->nb_files || n > SIZE_MAX / sizeof(*cat->files) ||
            !(new_files = av_realloc(cat->files, n * sizeof(*cat->files))))
            FAIL(AVERROR(ENOMEM));
        cat->files = new_files;
        *nb_files_alloc = n;
    }

    file = &cat->files[cat->nb_files++];
    memset(file, 0, sizeof(*file));
    *rfile = file;

    file->url        = url;
    file->start_time = AV_NOPTS_VALUE;
    file->duration   = AV_NOPTS_VALUE;

    file->inpoint    = AV_NOPTS_VALUE;
    file->outpoint   = AV_NOPTS_VALUE;

    return 0;

fail:
    av_free(url);
    av_free(filename);
    return ret;
}

























































































































































































































































































































































static int open_next_file(AVFormatContext *avf)
{
    ConcatContext *cat = avf->priv_data;
    unsigned fileno = cat->cur_file - cat->files;

    if (cat->cur_file->duration == AV_NOPTS_VALUE)
        cat->cur_file->duration = cat->avf->duration - (cat->cur_file->file_inpoint - cat->cur_file->file_start_time);







    if (++fileno >= cat->nb_files) {
        cat->eof = 1;
        return AVERROR_EOF;
    }
    return open_file(avf, fileno);
}


























/* Returns true if the packet dts is greater or equal to the specified outpoint. */









static int concat_read_packet(AVFormatContext *avf, AVPacket *pkt)
{
    ConcatContext *cat = avf->priv_data;
    int ret;
    int64_t delta;
    ConcatStream *cs;
    AVStream *st;

    if (cat->eof)
        return AVERROR_EOF;

    if (!cat->avf)
        return AVERROR(EIO);

    while (1) {
        ret = av_read_frame(cat->avf, pkt);
        if (ret == AVERROR_EOF) {
            if ((ret = open_next_file(avf)) < 0)
                return ret;
            continue;
        }
        if (ret < 0)
            return ret;
        if ((ret = match_streams(avf)) < 0) {
            av_packet_unref(pkt);
            return ret;
        }
        if (packet_after_outpoint(cat, pkt)) {
            av_packet_unref(pkt);
            if ((ret = open_next_file(avf)) < 0)
                return ret;
            continue;
        }
        cs = &cat->cur_file->streams[pkt->stream_index];
        if (cs->out_stream_index < 0) {
            av_packet_unref(pkt);
            continue;
        }
        pkt->stream_index = cs->out_stream_index;
        break;
    }
    if ((ret = filter_packet(avf, cs, pkt)))
        return ret;

    st = cat->avf->streams[pkt->stream_index];
    av_log(avf, AV_LOG_DEBUG, "file:%d stream:%d pts:%s pts_time:%s dts:%s dts_time:%s",
           (unsigned)(cat->cur_file - cat->files), pkt->stream_index,
           av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &st->time_base),
           av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, &st->time_base));

    delta = av_rescale_q(cat->cur_file->start_time - cat->cur_file->file_inpoint,
                         AV_TIME_BASE_Q,
                         cat->avf->streams[pkt->stream_index]->time_base);
    if (pkt->pts != AV_NOPTS_VALUE)
        pkt->pts += delta;
    if (pkt->dts != AV_NOPTS_VALUE)
        pkt->dts += delta;
    av_log(avf, AV_LOG_DEBUG, " -> pts:%s pts_time:%s dts:%s dts_time:%s\n",
           av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &st->time_base),
           av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, &st->time_base));
    if (cat->cur_file->metadata) {
        uint8_t* metadata;
        int metadata_len;
        char* packed_metadata = av_packet_pack_dictionary(cat->cur_file->metadata, &metadata_len);
        if (!packed_metadata)
            return AVERROR(ENOMEM);
        if (!(metadata = av_packet_new_side_data(pkt, AV_PKT_DATA_STRINGS_METADATA, metadata_len))) {
            av_freep(&packed_metadata);
            return AVERROR(ENOMEM);
        }
        memcpy(metadata, packed_metadata, metadata_len);
        av_freep(&packed_metadata);
    }








    return ret;
}





































































































#define OFFSET(x) offsetof(ConcatContext, x)
#define DEC AV_OPT_FLAG_DECODING_PARAM

static const AVOption options[] = {
    { "safe", "enable safe mode",
      OFFSET(safe), AV_OPT_TYPE_BOOL, {.i64 = 1}, -1, 1, DEC },
    { "auto_convert", "automatically convert bitstream format",
      OFFSET(auto_convert), AV_OPT_TYPE_BOOL, {.i64 = 1}, 0, 1, DEC },
    { "segment_time_metadata", "output file segment start time and duration as packet metadata",
      OFFSET(segment_time_metadata), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, DEC },
    { NULL }
};

static const AVClass concat_class = {
    .class_name = "concat demuxer",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};


AVInputFormat ff_concat_demuxer = {
    .name           = "concat",
    .long_name      = NULL_IF_CONFIG_SMALL("Virtual concatenation script"),
    .priv_data_size = sizeof(ConcatContext),
    .read_probe     = concat_probe,
    .read_header    = concat_read_header,
    .read_packet    = concat_read_packet,
    .read_close     = concat_read_close,
    .read_seek2     = concat_seek,
    .priv_class     = &concat_class,
};
