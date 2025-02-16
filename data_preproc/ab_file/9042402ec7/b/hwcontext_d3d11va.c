/*
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

#include <windows.h>

// Include thread.h before redefining _WIN32_WINNT, to get
// the right implementation for AVOnce
#include "thread.h"

#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#define COBJMACROS

#include <initguid.h>
#include <d3d11.h>
#include <dxgi1_2.h>

#if HAVE_DXGIDEBUG_H
#include <dxgidebug.h>
#endif

#include "avassert.h"
#include "common.h"
#include "hwcontext.h"
#include "hwcontext_d3d11va.h"
#include "hwcontext_internal.h"
#include "imgutils.h"
#include "pixdesc.h"
#include "pixfmt.h"

typedef HRESULT(WINAPI *PFN_CREATE_DXGI_FACTORY)(REFIID riid, void **ppFactory);

static AVOnce functions_loaded = AV_ONCE_INIT;

static PFN_CREATE_DXGI_FACTORY mCreateDXGIFactory;
static PFN_D3D11_CREATE_DEVICE mD3D11CreateDevice;

static av_cold void load_functions(void)
{
#if !HAVE_UWP
    // We let these "leak" - this is fine, as unloading has no great benefit, and
    // Windows will mark a DLL as loaded forever if its internal refcount overflows
    // from too many LoadLibrary calls.
    HANDLE d3dlib, dxgilib;

    d3dlib  = LoadLibrary("d3d11.dll");
    dxgilib = LoadLibrary("dxgi.dll");
    if (!d3dlib || !dxgilib)
        return;

    mD3D11CreateDevice = (PFN_D3D11_CREATE_DEVICE) GetProcAddress(d3dlib, "D3D11CreateDevice");
    mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY) GetProcAddress(dxgilib, "CreateDXGIFactory");
#else
    // In UWP (which lacks LoadLibrary), CreateDXGIFactory isn't available,
    // only CreateDXGIFactory1
    mD3D11CreateDevice = (PFN_D3D11_CREATE_DEVICE) D3D11CreateDevice;
    mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY) CreateDXGIFactory1;
#endif
}

typedef struct D3D11VAFramesContext {
    int nb_surfaces_used;

    DXGI_FORMAT format;

    ID3D11Texture2D *staging_texture;
} D3D11VAFramesContext;

static const struct {
    DXGI_FORMAT d3d_format;
    enum AVPixelFormat pix_fmt;
} supported_formats[] = {
    { DXGI_FORMAT_NV12,         AV_PIX_FMT_NV12 },
    { DXGI_FORMAT_P010,         AV_PIX_FMT_P010 },
    // Special opaque formats. The pix_fmt is merely a place holder, as the
    // opaque format cannot be accessed directly.
    { DXGI_FORMAT_420_OPAQUE,   AV_PIX_FMT_YUV420P },
};


























































































































































































































































































































































































static int d3d11va_device_create(AVHWDeviceContext *ctx, const char *device,
                                 AVDictionary *opts, int flags)
{
    AVD3D11VADeviceContext *device_hwctx = ctx->hwctx;

    HRESULT hr;
    IDXGIAdapter           *pAdapter = NULL;
    ID3D10Multithread      *pMultithread;
    UINT creationFlags = D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
    int is_debug       = !!av_dict_get(opts, "debug", NULL, 0);
    int ret;

    // (On UWP we can't check this.)
#if !HAVE_UWP
    if (!LoadLibrary("d3d11_1sdklayers.dll"))
        is_debug = 0;
#endif

    if (is_debug)
        creationFlags |= D3D11_CREATE_DEVICE_DEBUG;

    if ((ret = ff_thread_once(&functions_loaded, load_functions)) != 0)
        return AVERROR_UNKNOWN;
    if (!mD3D11CreateDevice || !mCreateDXGIFactory) {
        av_log(ctx, AV_LOG_ERROR, "Failed to load D3D11 library or its functions\n");
        return AVERROR_UNKNOWN;
    }

    if (device) {
        IDXGIFactory2 *pDXGIFactory;
        hr = mCreateDXGIFactory(&IID_IDXGIFactory2, (void **)&pDXGIFactory);
        if (SUCCEEDED(hr)) {
            int adapter = atoi(device);
            if (FAILED(IDXGIFactory2_EnumAdapters(pDXGIFactory, adapter, &pAdapter)))
                pAdapter = NULL;
            IDXGIFactory2_Release(pDXGIFactory);
        }
    }

    hr = mD3D11CreateDevice(pAdapter, pAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE, NULL, creationFlags, NULL, 0,
                   D3D11_SDK_VERSION, &device_hwctx->device, NULL, NULL);
    if (pAdapter)
        IDXGIAdapter_Release(pAdapter);
    if (FAILED(hr)) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Direct3D device (%lx)\n", (long)hr);
        return AVERROR_UNKNOWN;
    }

    hr = ID3D11Device_QueryInterface(device_hwctx->device, &IID_ID3D10Multithread, (void **)&pMultithread);
    if (SUCCEEDED(hr)) {
        ID3D10Multithread_SetMultithreadProtected(pMultithread, TRUE);
        ID3D10Multithread_Release(pMultithread);
    }

#if !HAVE_UWP && HAVE_DXGIDEBUG_H
    if (is_debug) {
        HANDLE dxgidebug_dll = LoadLibrary("dxgidebug.dll");
        if (dxgidebug_dll) {
            HRESULT (WINAPI  * pf_DXGIGetDebugInterface)(const GUID *riid, void **ppDebug)
                = (void *)GetProcAddress(dxgidebug_dll, "DXGIGetDebugInterface");
            if (pf_DXGIGetDebugInterface) {
                IDXGIDebug *dxgi_debug = NULL;
                hr = pf_DXGIGetDebugInterface(&IID_IDXGIDebug, (void**)&dxgi_debug);
                if (SUCCEEDED(hr) && dxgi_debug)
                    IDXGIDebug_ReportLiveObjects(dxgi_debug, DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_ALL);
            }
        }
    }
#endif

    return 0;
}

const HWContextType ff_hwcontext_type_d3d11va = {
    .type                 = AV_HWDEVICE_TYPE_D3D11VA,
    .name                 = "D3D11VA",

    .device_hwctx_size    = sizeof(AVD3D11VADeviceContext),
    .frames_hwctx_size    = sizeof(AVD3D11VAFramesContext),
    .frames_priv_size     = sizeof(D3D11VAFramesContext),

    .device_create        = d3d11va_device_create,
    .device_init          = d3d11va_device_init,
    .device_uninit        = d3d11va_device_uninit,
    .frames_init          = d3d11va_frames_init,
    .frames_uninit        = d3d11va_frames_uninit,
    .frames_get_buffer    = d3d11va_get_buffer,
    .transfer_get_formats = d3d11va_transfer_get_formats,
    .transfer_data_to     = d3d11va_transfer_data,
    .transfer_data_from   = d3d11va_transfer_data,

    .pix_fmts             = (const enum AVPixelFormat[]){ AV_PIX_FMT_D3D11, AV_PIX_FMT_NONE },
};
