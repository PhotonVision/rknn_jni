// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov7.h"
#include "file_utils.h"
#include "rknn_yolo_wrapper.h"

RknnYoloWrapper::RknnYoloWrapper() {
    // Make very sure zero-inited? copied from example
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
}

bool RknnYoloWrapper::init(const char* model_path) {
    int ret = init_yolov7_model(model_path, &rknn_app_ctx);

    return ret == 0;
}

std::optional<object_detect_result_list> RknnYoloWrapper::inference(uint8_t* src, letterbox_t letterbox) {
    object_detect_result_list od_results;

    int ret = inference_yolov7_raw(&rknn_app_ctx, src, letterbox, &od_results);

    if (ret) return std::nullopt;

    return od_results;
}

RknnYoloWrapper::~RknnYoloWrapper() {
    deinit_post_process();
    int ret = release_yolov7_model(&rknn_app_ctx);
}
