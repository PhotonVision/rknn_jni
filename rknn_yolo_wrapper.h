#pragma once

#include "yolov7.h"
#include "postprocess.h"
#include <optional>

class RknnYoloWrapper {
private:
    rknn_app_context_t rknn_app_ctx {0};

public:
    RknnYoloWrapper();
    bool init(const char* model_path);
    std::optional<object_detect_result_list> inference(uint8_t* src, letterbox_t letterbox);
    ~RknnYoloWrapper();
};