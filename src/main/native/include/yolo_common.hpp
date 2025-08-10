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

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "rknn_api.h"
#include "yolov5/postprocess_v5.h"

typedef struct {
  double nms_thresh;
  double box_thresh;
} DetectionFilterParams;

enum ModelVersion { YOLO_V5, YOLO_V8, YOLO_V11 };

class YoloModel {
public:
  YoloModel(std::string modelPath, int num_classes_, ModelVersion type_,
            int coreNumber);

  int changeCoreMask(int coreNumber);

  detect_result_group_t forward(cv::Mat &orig_img,
                                DetectionFilterParams params);

  ~YoloModel();

  bool is_quant;
protected:
  virtual detect_result_group_t postProcess(std::vector<rknn_output> output,
                                            DetectionFilterParams params,
                                            cv::Size inputImageSize,
                                            cv::Size2d inputImageScale,
                                            BOX_RECT letterbox) = 0;

  int numClasses;
  ModelVersion yoloType;

  // todo matt do we need to keep this model data pointer around? seems like no?
  unsigned char *model_data;
  rknn_context ctx;
  cv::Size modelSize;
  int channels;

  rknn_input_output_num io_num;
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;

  rknn_input inputs[1];
};

class YoloV5Model : public YoloModel {
public:
  YoloV5Model(std::string modelPath, int num_classes_, int coreNumber)
      : YoloModel(modelPath, num_classes_, ModelVersion::YOLO_V5, coreNumber) {}

protected:
  detect_result_group_t postProcess(std::vector<rknn_output> output,
                                    DetectionFilterParams params,
                                    cv::Size inputImageSize,
                                    cv::Size2d inputImageScale,
                                    BOX_RECT letterbox);
};

class YoloV8Model : public YoloModel {
public:
  YoloV8Model(std::string modelPath, int num_classes_, int coreNumber)
      : YoloModel(modelPath, num_classes_, ModelVersion::YOLO_V8, coreNumber) {}

protected:
  detect_result_group_t postProcess(std::vector<rknn_output> output,
                                    DetectionFilterParams params,
                                    cv::Size inputImageSize,
                                    cv::Size2d inputImageScale,
                                    BOX_RECT letterbox);
};

class YoloV11Model : public YoloModel {
public:
  YoloV11Model(std::string modelPath, int num_classes_, int coreNumber)
      : YoloModel(modelPath, num_classes_, ModelVersion::YOLO_V11, coreNumber) {
  }

protected:
  detect_result_group_t postProcess(std::vector<rknn_output> output,
                                    DetectionFilterParams params,
                                    cv::Size inputImageSize,
                                    cv::Size2d inputImageScale,
                                    BOX_RECT letterbox);
};
