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

#ifndef RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_MODEL_ZOO_COMMON_H_
#define RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_MODEL_ZOO_COMMON_H_

/**
 * @brief Image pixel format
 *
 */
typedef enum {
  IMAGE_FORMAT_GRAY8,
  IMAGE_FORMAT_RGB888,
  IMAGE_FORMAT_RGBA8888,
  IMAGE_FORMAT_YUV420SP_NV21,
  IMAGE_FORMAT_YUV420SP_NV12,
} image_format_t;

/**
 * @brief Image buffer
 *
 */
typedef struct {
  int width;
  int height;
  int width_stride;
  int height_stride;
  image_format_t format;
  unsigned char *virt_addr;
  int size;
  int fd;
} image_buffer_t;

/**
 * @brief Image rectangle
 *
 */
typedef struct {
  int left;
  int top;
  int right;
  int bottom;
} image_rect_t;

#endif // RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_MODEL_ZOO_COMMON_H_
