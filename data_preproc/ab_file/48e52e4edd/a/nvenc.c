/*
 * H.264/HEVC hardware encoding using nvidia nvenc
 * Copyright (c) 2016 Timo Rothenpieler <timo@rothenpieler.org>
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

#include "nvenc.h"

#include "libavutil/hwcontext_cuda.h"
#include "libavutil/hwcontext.h"
#include "libavutil/imgutils.h"
#include "libavutil/avassert.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"
#include "internal.h"

#define NVENC_CAP 0x30
#define IS_CBR(rc) (rc == NV_ENC_PARAMS_RC_CBR ||             \
                    rc == NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ || \
                    rc == NV_ENC_PARAMS_RC_CBR_HQ)

const enum AVPixelFormat ff_nvenc_pix_fmts[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_0RGB32,
    AV_PIX_FMT_0BGR32,
    AV_PIX_FMT_CUDA,
#if CONFIG_D3D11VA
    AV_PIX_FMT_D3D11,
#endif
    AV_PIX_FMT_NONE
};

#define IS_10BIT(pix_fmt)  (pix_fmt == AV_PIX_FMT_P010    || \
                            pix_fmt == AV_PIX_FMT_YUV444P16)

#define IS_YUV444(pix_fmt) (pix_fmt == AV_PIX_FMT_YUV444P || \
                            pix_fmt == AV_PIX_FMT_YUV444P16)

static const struct {
    NVENCSTATUS nverr;
    int         averr;
    const char *desc;
} nvenc_errors[] = {
    { NV_ENC_SUCCESS,                      0,                "success"                  },
    { NV_ENC_ERR_NO_ENCODE_DEVICE,         AVERROR(ENOENT),  "no encode device"         },
    { NV_ENC_ERR_UNSUPPORTED_DEVICE,       AVERROR(ENOSYS),  "unsupported device"       },
    { NV_ENC_ERR_INVALID_ENCODERDEVICE,    AVERROR(EINVAL),  "invalid encoder device"   },
    { NV_ENC_ERR_INVALID_DEVICE,           AVERROR(EINVAL),  "invalid device"           },
    { NV_ENC_ERR_DEVICE_NOT_EXIST,         AVERROR(EIO),     "device does not exist"    },
    { NV_ENC_ERR_INVALID_PTR,              AVERROR(EFAULT),  "invalid ptr"              },
    { NV_ENC_ERR_INVALID_EVENT,            AVERROR(EINVAL),  "invalid event"            },
    { NV_ENC_ERR_INVALID_PARAM,            AVERROR(EINVAL),  "invalid param"            },
    { NV_ENC_ERR_INVALID_CALL,             AVERROR(EINVAL),  "invalid call"             },
    { NV_ENC_ERR_OUT_OF_MEMORY,            AVERROR(ENOMEM),  "out of memory"            },
    { NV_ENC_ERR_ENCODER_NOT_INITIALIZED,  AVERROR(EINVAL),  "encoder not initialized"  },
    { NV_ENC_ERR_UNSUPPORTED_PARAM,        AVERROR(ENOSYS),  "unsupported param"        },
    { NV_ENC_ERR_LOCK_BUSY,                AVERROR(EAGAIN),  "lock busy"                },
    { NV_ENC_ERR_NOT_ENOUGH_BUFFER,        AVERROR_BUFFER_TOO_SMALL, "not enough buffer"},
    { NV_ENC_ERR_INVALID_VERSION,          AVERROR(EINVAL),  "invalid version"          },
    { NV_ENC_ERR_MAP_FAILED,               AVERROR(EIO),     "map failed"               },
    { NV_ENC_ERR_NEED_MORE_INPUT,          AVERROR(EAGAIN),  "need more input"          },
    { NV_ENC_ERR_ENCODER_BUSY,             AVERROR(EAGAIN),  "encoder busy"             },
    { NV_ENC_ERR_EVENT_NOT_REGISTERD,      AVERROR(EBADF),   "event not registered"     },
    { NV_ENC_ERR_GENERIC,                  AVERROR_UNKNOWN,  "generic error"            },
    { NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY,  AVERROR(EINVAL),  "incompatible client key"  },
    { NV_ENC_ERR_UNIMPLEMENTED,            AVERROR(ENOSYS),  "unimplemented"            },
    { NV_ENC_ERR_RESOURCE_REGISTER_FAILED, AVERROR(EIO),     "resource register failed" },
    { NV_ENC_ERR_RESOURCE_NOT_REGISTERED,  AVERROR(EBADF),   "resource not registered"  },
    { NV_ENC_ERR_RESOURCE_NOT_MAPPED,      AVERROR(EBADF),   "resource not mapped"      },
};






























































































































































































































































































































































































































































































typedef struct GUIDTuple {
    const GUID guid;
    int flags;
} GUIDTuple;

#define PRESET_ALIAS(alias, name, ...) \
    [PRESET_ ## alias] = { NV_ENC_PRESET_ ## name ## _GUID, __VA_ARGS__ }

#define PRESET(name, ...) PRESET_ALIAS(name, name, __VA_ARGS__)
























#undef PRESET
#undef PRESET_ALIAS


























































































































































































































































































































































































































































































































































































































































































































































































































































































































































static int nvenc_find_free_reg_resource(AVCodecContext *avctx)
{
    NvencContext *ctx = avctx->priv_data;
    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;
    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;


    int i;

    if (ctx->nb_registered_frames == FF_ARRAY_ELEMS(ctx->registered_frames)) {
        for (i = 0; i < ctx->nb_registered_frames; i++) {
            if (!ctx->registered_frames[i].mapped) {
                if (ctx->registered_frames[i].regptr) {
                    p_nvenc->nvEncUnregisterResource(ctx->nvencoder,
                                                ctx->registered_frames[i].regptr);

                    ctx->registered_frames[i].regptr = NULL;
                }
                return i;
            }
        }
    } else {
        return ctx->nb_registered_frames++;
    }

    av_log(avctx, AV_LOG_ERROR, "Too many registered CUDA frames\n");
    return AVERROR(ENOMEM);
}


























































































































































































static int process_output_surface(AVCodecContext *avctx, AVPacket *pkt, NvencSurface *tmpoutsurf)
{
    NvencContext *ctx = avctx->priv_data;
    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;
    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;

    uint32_t slice_mode_data;
    uint32_t *slice_offsets = NULL;
    NV_ENC_LOCK_BITSTREAM lock_params = { 0 };
    NVENCSTATUS nv_status;
    int res = 0;

    enum AVPictureType pict_type;

    switch (avctx->codec->id) {
    case AV_CODEC_ID_H264:
      slice_mode_data = ctx->encode_config.encodeCodecConfig.h264Config.sliceModeData;
      break;
    case AV_CODEC_ID_H265:
      slice_mode_data = ctx->encode_config.encodeCodecConfig.hevcConfig.sliceModeData;
      break;
    default:
      av_log(avctx, AV_LOG_ERROR, "Unknown codec name\n");
      res = AVERROR(EINVAL);
      goto error;
    }
    slice_offsets = av_mallocz(slice_mode_data * sizeof(*slice_offsets));

    if (!slice_offsets) {
        res = AVERROR(ENOMEM);
        goto error;
    }

    lock_params.version = NV_ENC_LOCK_BITSTREAM_VER;

    lock_params.doNotWait = 0;
    lock_params.outputBitstream = tmpoutsurf->output_surface;
    lock_params.sliceOffsets = slice_offsets;

    nv_status = p_nvenc->nvEncLockBitstream(ctx->nvencoder, &lock_params);
    if (nv_status != NV_ENC_SUCCESS) {
        res = nvenc_print_error(avctx, nv_status, "Failed locking bitstream buffer");
        goto error;
    }

    if (res = ff_alloc_packet2(avctx, pkt, lock_params.bitstreamSizeInBytes,0)) {
        p_nvenc->nvEncUnlockBitstream(ctx->nvencoder, tmpoutsurf->output_surface);
        goto error;
    }

    memcpy(pkt->data, lock_params.bitstreamBufferPtr, lock_params.bitstreamSizeInBytes);

    nv_status = p_nvenc->nvEncUnlockBitstream(ctx->nvencoder, tmpoutsurf->output_surface);
    if (nv_status != NV_ENC_SUCCESS)
        nvenc_print_error(avctx, nv_status, "Failed unlocking bitstream buffer, expect the gates of mordor to open");




    if (avctx->pix_fmt == AV_PIX_FMT_CUDA || avctx->pix_fmt == AV_PIX_FMT_D3D11) {
        ctx->registered_frames[tmpoutsurf->reg_idx].mapped -= 1;
        if (ctx->registered_frames[tmpoutsurf->reg_idx].mapped == 0) {
            p_nvenc->nvEncUnmapInputResource(ctx->nvencoder, ctx->registered_frames[tmpoutsurf->reg_idx].in_map.mappedResource);
            p_nvenc->nvEncUnregisterResource(ctx->nvencoder, ctx->registered_frames[tmpoutsurf->reg_idx].regptr);








            ctx->registered_frames[tmpoutsurf->reg_idx].regptr = NULL;
        } else if (ctx->registered_frames[tmpoutsurf->reg_idx].mapped < 0) {
            res = AVERROR_BUG;
            goto error;
        }

        av_frame_unref(tmpoutsurf->in_ref);

        tmpoutsurf->input_surface = NULL;
    }

    switch (lock_params.pictureType) {
    case NV_ENC_PIC_TYPE_IDR:
        pkt->flags |= AV_PKT_FLAG_KEY;
    case NV_ENC_PIC_TYPE_I:
        pict_type = AV_PICTURE_TYPE_I;
        break;
    case NV_ENC_PIC_TYPE_P:
        pict_type = AV_PICTURE_TYPE_P;
        break;
    case NV_ENC_PIC_TYPE_B:
        pict_type = AV_PICTURE_TYPE_B;
        break;
    case NV_ENC_PIC_TYPE_BI:
        pict_type = AV_PICTURE_TYPE_BI;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Unknown picture type encountered, expect the output to be broken.\n");
        av_log(avctx, AV_LOG_ERROR, "Please report this error and include as much information on how to reproduce it as possible.\n");
        res = AVERROR_EXTERNAL;
        goto error;
    }

#if FF_API_CODED_FRAME
FF_DISABLE_DEPRECATION_WARNINGS
    avctx->coded_frame->pict_type = pict_type;
FF_ENABLE_DEPRECATION_WARNINGS
#endif

    ff_side_data_set_encoder_stats(pkt,
        (lock_params.frameAvgQP - 1) * FF_QP2LAMBDA, NULL, 0, pict_type);

    res = nvenc_set_timestamp(avctx, &lock_params, pkt);
    if (res < 0)
        goto error2;

    av_free(slice_offsets);

    return 0;

error:
    timestamp_queue_dequeue(ctx->timestamp_list);

error2:
    av_free(slice_offsets);

    return res;
}





















































































































































































