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

#include <iostream>
#include <vector>

#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void checkOpenCL() {
  if (!cv::ocl::haveOpenCL()) {
    std::cout << "OpenCL is not available..." << std::endl;
    // return;
  } else {
    std::cout << "Opencl claims to exist";
  }

  std::cout << cv::getBuildInformation() << std::endl;

  cv::ocl::Context context;
  if (!context.create(cv::ocl::Device::TYPE_ALL)) {
    std::cout << "Failed creating the context..." << std::endl;
    // return;
  }

  std::cout << context.ndevices() << " CPU devices are detected."
            << std::endl; // This bit provides an overview of the OpenCL devices
                          // you have in your computer
  for (int i = 0; i < context.ndevices(); i++) {
    cv::ocl::Device device = context.device(i);
    std::cout << "name:              " << device.name() << std::endl;
    std::cout << "available:         " << device.available() << std::endl;
    std::cout << "imageSupport:      " << device.imageSupport() << std::endl;
    std::cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version()
              << std::endl;
    std::cout << std::endl;
  }

  cv::ocl::Device(context.device(
      0)); // Here is where you change which GPU to use (e.g. 0 or 1)
}

#include <chrono>

class Timer {
public:
  Timer() : beg_(clock_::now()) {}
  void reset() { beg_ = clock_::now(); }
  double elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(clock_::now() -
                                                                 beg_)
        .count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1>> second_;
  std::chrono::time_point<clock_> beg_;
};

void trydnn() {
  Timer timer;
  using cv::Mat;
  using cv::dnn::readNetFromDarknet;
  using std::vector;
  auto net = readNetFromDarknet(
      "/home/coolpi/rknn_java/opencv_test/yolov4-csp-swish.cfg",
      "/home/coolpi/rknn_java/opencv_test/yolov4-csp-swish.weights");

  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_OPENCL);

  cout << "got names\n";

  vector<String> names = net.getUnconnectedOutLayersNames();
  for (const auto &n : names)
    cout << n << " ";
  cout << endl;

  auto img = cv::imread("../../src/test/resources/bus.jpg");
  auto blob = blobFromImage(img, 1.0 / 255.0, {640, 640});

  // vector<String> names = {
  //     "yolo_167", "yolo_171", "yolo_175"};

  // vector<int> outLayers = net.getUnconnectedOutLayers();
  // auto layersNames = net.getLayerNames();
  // vector<String> names;
  // for (const auto& o : outLayers) {
  //     names.push_back(layersNames[o - 1]);
  // }

  int BOUND = 15;
  int total = 0;
  for (int i = 0; i < BOUND; i++) {
    timer.reset();
    cout << "Setting inputs\n";
    net.setInput(blob);

    cout << "Forward\n";
    vector<Mat> outs;
    net.forward(outs, names);

    std::cout << "cycle took: " << timer.elapsed() << "ms" << endl;
    total += timer.elapsed();
  }
  total /= BOUND;
  cout << "Mean: " << total << "ms\n";
}

int main() {
  checkOpenCL();

  trydnn();

  return 0;
}
