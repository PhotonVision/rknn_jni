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

#ifndef RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_YOLOV5_POSTPROCESS_V5_H_
#define RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_YOLOV5_POSTPROCESS_V5_H_

#include <stdint.h>

#include <vector>

#define OBJ_NAME_MAX_SIZE_V5 16
#define OBJ_NUMB_MAX_SIZE_V5 64
// #define OBJ_CLASS_NUM 80
// #define NMS_THRESH 0.45
// #define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + numClasses)

typedef struct _BOX_RECT {
  int left;
  int right;
  int top;
  int bottom;
} BOX_RECT;

typedef struct __detect_result_t {
  int id;
  BOX_RECT box;
  float obj_conf;
} detect_result_t;

typedef struct _detect_result_group_t {
  int id;
  int count;
  std::vector<detect_result_t> results;
} detect_result_group_t;

int post_process_v5(int8_t *input0, int8_t *input1, int8_t *input2,
                    int model_in_h, int model_in_w, float conf_threshold,
                    float nms_threshold, BOX_RECT pads, float scale_w,
                    float scale_h, std::vector<int32_t> &qnt_zps,
                    std::vector<float> &qnt_scales,
                    detect_result_group_t *group, int numClasses);

#endif // RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_YOLOV5_POSTPROCESS_V5_H_
