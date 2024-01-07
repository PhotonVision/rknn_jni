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

#ifndef RKNN_JAVA_YOLOV7_H_
#define RKNN_JAVA_YOLOV7_H_

#include "common.h"
#include "rknn_api.h"

typedef struct {
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr *input_attrs;
  rknn_tensor_attr *output_attrs;
  int model_channel;
  int model_width;
  int model_height;
  bool is_quant;
} rknn_app_context_t;

#include "postprocess.h"

int init_yolov7_model(const char *model_path, rknn_app_context_t *app_ctx);

int release_yolov7_model(rknn_app_context_t *app_ctx);

int inference_yolov7_model(rknn_app_context_t *app_ctx, image_buffer_t *img,
                           object_detect_result_list *od_results);

/**
 * Run inference without bothering with RGA letterboxing, since I'm lazy
 * @param app_ctx App ctx from above
 * @param image_ptr Pointer to NCHW data blob
 * @param letterbox the letterbox applied to crop the image
 * @param od_results output result array
 */
int inference_yolov7_raw(rknn_app_context_t *app_ctx, uint8_t *image_ptr,
                         letterbox_t letterbox,
                         object_detect_result_list *od_results);

#endif // RKNN_JAVA_YOLOV7_H_
