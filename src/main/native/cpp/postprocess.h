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

#ifndef RKNN_JAVA_SRC_MAIN_NATIVE_CPP_POSTPROCESS_H_
#define RKNN_JAVA_SRC_MAIN_NATIVE_CPP_POSTPROCESS_H_

#include <stdint.h>

#include <vector>

#include "common.h"
#include "rknn_api.h"
#include "yolov7.h"

int init_post_process();
char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list *od_results);

void deinitPostProcess();
#endif // RKNN_JAVA_SRC_MAIN_NATIVE_CPP_POSTPROCESS_H_
