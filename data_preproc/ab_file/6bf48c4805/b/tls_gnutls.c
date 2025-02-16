/*
 * TLS/SSL Protocol
 * Copyright (c) 2011 Martin Storsjo
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

#include <errno.h>

#include <gnutls/gnutls.h>
#include <gnutls/x509.h>

#include "avformat.h"
#include "internal.h"
#include "network.h"
#include "os_support.h"
#include "url.h"
#include "tls.h"
#include "libavcodec/internal.h"
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"

#ifndef GNUTLS_VERSION_NUMBER
#define GNUTLS_VERSION_NUMBER LIBGNUTLS_VERSION_NUMBER
#endif

#if HAVE_THREADS && GNUTLS_VERSION_NUMBER <= 0x020b00
#include <gcrypt.h>
#include "libavutil/thread.h"
GCRY_THREAD_OPTION_PTHREAD_IMPL;
#endif

typedef struct TLSContext {
    const AVClass *class;
    TLSShared tls_shared;
    gnutls_session_t session;
    gnutls_certificate_credentials_t cred;
    int need_shutdown;
} TLSContext;




































































































































































































static const AVOption options[] = {
    TLS_COMMON_OPTIONS(TLSContext, tls_shared),
    { NULL }
};

static const AVClass tls_class = {
    .class_name = "tls",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const URLProtocol ff_tls_gnutls_protocol = {
    .name           = "tls",
    .url_open2      = tls_open,
    .url_read       = tls_read,
    .url_write      = tls_write,
    .url_close      = tls_close,
    .url_get_file_handle = tls_get_file_handle,
    .priv_data_size = sizeof(TLSContext),
    .flags          = URL_PROTOCOL_FLAG_NETWORK,
    .priv_data_class = &tls_class,
};
