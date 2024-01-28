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


#ifndef _RKNN_DEMO_YOLOV8_H_
#define _RKNN_DEMO_YOLOV8_H_

#include "rknn_api.h"
#include "common.h"
#include "postprocess_v8.h"

#include "opencv2/core/core.hpp"

class rkYolov8s
{
private:
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;


public:
    // rkYolov5s(const std::string &model_path, int numClasses);

    int init_yolov8_model(const char* model_path, rknn_app_context_t* app_ctx);

    int init(const char* model_path) {
        return init_yolov8_model(model_path, &context);
    }

    // rknn_context *get_pctx();

    /**
     * Run forward inference only, returning resulting detections
    */
    // int inferOnly(cv::Mat &orig_img, detect_result_group_t *outReults, DetectionFilterParams params);

    int inference_yolov8_model(rknn_app_context_t *app_ctx, cv::Mat &orig_img, detect_result_group_t *od_results, DetectionFilterParams params);
    int infer(cv::Mat &orig_img, detect_result_group_t *od_results, DetectionFilterParams params) {
        return inference_yolov8_model(&context, orig_img, od_results, params);
    }

    int release_yolov8_model();

    ~rkYolov8s() {
        release_yolov8_model(&context);
    }
};

#endif //_RKNN_DEMO_YOLOV8_H_