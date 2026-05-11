/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_RGA_FUNC_H_
#define RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_RGA_FUNC_H_

#include <dlfcn.h>

#include "RgaApi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*FUNC_RGA_INIT)();
typedef void (*FUNC_RGA_DEINIT)();
typedef int (*FUNC_RGA_BLIT)(rga_info_t *, rga_info_t *, rga_info_t *);

typedef struct _rga_context {
  void *rga_handle;
  FUNC_RGA_INIT init_func;
  FUNC_RGA_DEINIT deinit_func;
  FUNC_RGA_BLIT blit_func;
} rga_context;

int RGA_init(rga_context *rga_ctx);

void img_resize_fast(rga_context *rga_ctx, int src_fd, int src_w, int src_h,
                     uint64_t dst_phys, int dst_w, int dst_h);

void img_resize_slow(rga_context *rga_ctx, void *src_virt, int src_w, int src_h,
                     void *dst_virt, int dst_w, int dst_h);

int RGA_deinit(rga_context *rga_ctx);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_RGA_FUNC_H_
