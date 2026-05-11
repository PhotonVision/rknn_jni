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

#ifndef RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_YOLOV8_11_POSTPROCESS_V8_11_H_
#define RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_YOLOV8_11_POSTPROCESS_V8_11_H_

#include <stdint.h>

#include <vector>

#include <opencv2/core/types.hpp>

#include "common.h"
#include "rknn_api.h"
#include "yolov5/postprocess_v5.h"

#define OBJ_NAME_MAX_SIZE_V8_11 64
#define OBJ_NUMB_MAX_SIZE_V8_11 128
// #define OBJ_CLASS_NUM 3
// #define NMS_THRESH 0.45
// #define BOX_THRESH 0.25

// int init_post_process();
// void deinit_post_process();
// char *coco_cls_to_name(int cls_id);
int post_process_v8_11(cv::Size modelSize, rknn_output *outputs,
                       BOX_RECT *letter_box, float conf_threshold,
                       float nms_threshold, detect_result_group_t *od_results,
                       int numClasses,
                       std::vector<rknn_tensor_attr> &output_attrs,
                       bool is_quant, int n_outputs);

#endif // RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_YOLOV8_11_POSTPROCESS_V8_11_H_
