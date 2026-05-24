#include "yolo_common.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main_test(ModelVersion version)
{

    YoloModel *wrapper;

    if (version == ModelVersion::YOLO_V5) {
        printf("Starting with version 5\n");
        wrapper = new YoloV5Model("note-640-640-yolov5s.rknn", 1, 0);
    } else if (version == ModelVersion::YOLO_V8) {
        printf("Starting with version 8\n");
        wrapper = new YoloV8Model("note-robot-yolov8s-quant.rknn", 3, 0);
    } else if (version == ModelVersion::YOLO_V11){
        printf("Starting with version 11\n");
        wrapper = new YoloV11Model("note-robot-yolov11s-quant.rknn", 3, 0);
    } else if (version == ModelVersion::YOLO_V11OBB){
        printf("Starting with version 11OBB\n");
        wrapper = new YoloV11OBBModel("cageconverted-640-640-yolov11obbn.rknn", 1, 0);
        // wrapper = new YoloV11OBBModel("best_yolo11obbn_cage_CONVERTED_FP32.rknn", 1, 0);
        // wrapper = new YoloV11OBBModel("epochbestREAL-1024-1024-yolov11obbm.rknn", 2, 0);
    } else {
        printf("Unknown version\n");
        return 1;
    }

    std::cout << "created: " << (long unsigned int)wrapper << std::endl;

    for (int j = 0; j < 1; j++) {
        cv::Mat img;
        img = cv::imread("test_cage_2_rotated.jpg");

        // cv::resize(img, img, cv::Size(1024, 1024));

        DetectionFilterParams params {
            .nms_thresh = 0.45,
            .box_thresh = 0.15,
        };
        auto ret = wrapper->forward(img, params);

        std::cout << "Count: " << ret.count << std::endl;
        for (int i = 0; i < ret.count; ++i) {
            std::cout << "ID: " << ret.results[i].id << " conf " << ret.results[i].obj_conf << " @ "
                << "cx: " << ret.results[i].obb.cx << " - "
                << "cy: " << ret.results[i].obb.cy << " - "
                << "width: " << ret.results[i].obb.width << " - "
                << "height: " << ret.results[i].obb.height << " - "
                << "angle: " << ret.results[i].obb.angle << " - "
                << std::endl;

            auto *det_result = &(ret.results[i]);

            int cx = det_result->obb.cx;
            int cy = det_result->obb.cy;
            int w  = det_result->obb.width;
            int h  = det_result->obb.height;
            float angle_rad = det_result->obb.angle;
            float angle_deg = angle_rad * 180.0f / CV_PI; // RotatedRect expects degrees

            cv::RotatedRect rrect(cv::Point2f((float)cx, (float)cy), cv::Size2f((float)w, (float)h), angle_deg);

            cv::Point2f pts2f[4];
            rrect.points(pts2f); // fill 4 corners

            std::vector<cv::Point> box_pts(4);
            for (int k = 0; k < 4; ++k) box_pts[k] = pts2f[k]; // round to ints

            std::vector<std::vector<cv::Point>> contour(1, box_pts);
            cv::drawContours(img, contour, 0, cv::Scalar(0, 255, 0), 3);

            // place label near top left of the rotated box
            cv::Rect bbox = cv::boundingRect(box_pts);

            char text[256];
            sprintf(text, "id%i %.1f%%", det_result->id, det_result->obj_conf * 100);
            cv::putText(img, text, {bbox.x, bbox.y + 10}, cv::FONT_ITALIC, 1.25, {100, 255, 0});
        }

        cv::imwrite("out2.png", img);
    }

    std::cout << "Deleting: " << (long unsigned int)wrapper << std::endl;
    delete wrapper;

    return 0;
}

#include <thread>
int main() {
//     std::vector<std::thread> threads;
// for (int i=1; i<=1; ++i)
//     threads.emplace_back(std::thread([]() {main_test("../note-640-640-yolov5s.rknn");}));
//     for (auto& th : threads) th.join();

    main_test(ModelVersion::YOLO_V11OBB);
    // main_test(ModelVersion::YOLO_V11);
    // main_test(ModelVersion::YOLO_V8);
    // main_test(ModelVersion::YOLO_V5);

    return 0;
}
