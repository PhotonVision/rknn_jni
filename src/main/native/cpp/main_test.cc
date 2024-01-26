#include "rknn_wrapper.h"

int main_test(const char* model_name)
{

    RknnWrapper *wrapper  = new RknnWrapper(model_name, 1);
    std::cout << "created: " << (long unsigned int)wrapper << std::endl;

    for (int i = 0; i < 20; i++) {
        cv::Mat img;
        img = cv::imread("../src/test/resources/bus.jpg");

        DetectionFilterParams params {
            .nms_thresh = 0.45,
            .box_thresh = 0.25,
        };
        auto ret = wrapper->forward(img, params);

        std::cout << "Count: " << ret.count << std::endl;
    }

    std::cout << "Deleting: " << (long unsigned int)wrapper << std::endl;
    delete wrapper;

    return 0;
}

#include <thread>
int main() {
    std::vector<std::thread> threads;
for (int i=1; i<=10; ++i)
    threads.emplace_back(std::thread([]() {main_test("../note-640-640-yolov5s.rknn");}));
    for (auto& th : threads) th.join();
}
