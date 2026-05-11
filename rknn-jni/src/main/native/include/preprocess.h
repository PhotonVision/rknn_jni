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

#ifndef RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_PREPROCESS_H_
#define RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_PREPROCESS_H_

#include <cstdio>

#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rga.h"

void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads,
               const float scale, const cv::Size &target_size,
               const cv::Scalar &pad_color = cv::Scalar(128, 128, 128));

int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image,
               cv::Mat &resized_image, const cv::Size &target_size);

#endif // RKNN_JNI_SRC_MAIN_NATIVE_INCLUDE_PREPROCESS_H_
