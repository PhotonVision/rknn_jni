#include "rknn_wrapper.h"

int main(int argc, char **argv)
{
    char *model_name = NULL;
    if (argc != 2)
    {
        printf("Usage: %s <rknn model> \n", argv[0]);
        return -1;
    }
    // Path to model
    model_name = (char *)argv[1];

    cv::namedWindow("Camera FPS");

    RknnWrapper wrapper (model_name, 3);

    for (int i = 0; i < 20; i++) {
        cv::Mat img;
        img = cv::imread("src/test/resources/bus.jpg");

        if (!wrapper.EnqueueMat(&img)) {
            break;
        }
        if (!wrapper.DequeueMat(&img) && i > 3) {
            break;
        }
        std::cout << "Got mat! " << img.data() << std::endl;
        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            break;
    }
    return 0;
}
