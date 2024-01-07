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

#ifndef RKNN_JAVA_UTILS_COMMON_H_
#define RKNN_JAVA_UTILS_COMMON_H_

// Common definitions for thresholds and max stuffs
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_NAME_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

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

typedef struct {
  int x_pad;
  int y_pad;
  float scale;
} letterbox_t;

typedef struct {
  image_rect_t box;
  float prop;
  int cls_id;
} object_detect_result;

typedef struct {
  int id;
  int count; // Number of results (ew)
  object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

#endif // RKNN_JAVA_UTILS_COMMON_H_
