#include <iostream>

#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void checkOpenCL()
{
    using namespace std;

    if (!cv::ocl::haveOpenCL())
    {
        cout << "OpenCL is not available..." << endl;
        // return;
    }
    else
    {
        cout << "Opencl claims to exist";
    }

    std::cout << cv::getBuildInformation() << std::endl;

    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_ALL))
    {
        cout << "Failed creating the context..." << endl;
        // return;
    }

    cout << context.ndevices() << " CPU devices are detected." << endl; // This bit provides an overview of the OpenCL devices you have in your computer
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        cout << "name:              " << device.name() << endl;
        cout << "available:         " << device.available() << endl;
        cout << "imageSupport:      " << device.imageSupport() << endl;
        cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << endl;
        cout << endl;
    }

    cv::ocl::Device(context.device(0)); // Here is where you change which GPU to use (e.g. 0 or 1)
}

#include <chrono>

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<std::chrono::milliseconds>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

void trydnn()
{
    Timer timer;

    using namespace std;
    using namespace cv::dnn;
    using namespace cv;
    auto net = readNetFromDarknet(
        "/home/matt/Downloads/yolov4-csp-swish.cfg",
        "/home/matt/Downloads/yolov4-csp-swish.weights");

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_OPENCL);


    auto img = cv::imread("../src/test/resources/bus.jpg");
    auto blob = blobFromImage(img, 1.0 / 255.0, {640, 640});

    cout << "got names\n";

    std::vector<String> names = net.getUnconnectedOutLayersNames();
    // std::vector<String> names = {
    //     "yolo_167", "yolo_171", "yolo_175"};

    // std::vector<int> outLayers = net.getUnconnectedOutLayers();
    // auto layersNames = net.getLayerNames();
    // std::vector<String> names;
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
        std::vector<Mat> outs;
        net.forward(outs, names);

        std::cout << "cycle took: " << timer.elapsed() << "ms" << endl;
        total += timer.elapsed();
    }
    total /= BOUND;
    cout << "Mean: " << total << "ms\n";
}

int main()
{
    checkOpenCL();

    trydnn();

    return 0;
}
