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

#include <optional>

#include "postprocess.h"
#include "yolov7.h"

class RknnYoloWrapper {
private:
  rknn_app_context_t rknn_app_ctx{0};

public:
  RknnYoloWrapper();
  bool init(const char *model_path);
  std::optional<object_detect_result_list> inference(uint8_t *src,
                                                     letterbox_t letterbox);
  ~RknnYoloWrapper();
};
