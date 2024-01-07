#include "rknn_wrapper.h"

int main(int argc, char **argv)
{
    char *model_name = NULL;
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
        return -1;
    }
    // 参数二，模型所在路径/The path where the model is located
    model_name = (char *)argv[1];
    // 参数三, 视频/摄像头
    char *vedio_name = argv[2];

    cv::namedWindow("Camera FPS");
    cv::VideoCapture capture;
    if (strlen(vedio_name) == 1)
        capture.open((int)(vedio_name[0] - '0'));
    else
        capture.open(vedio_name);

    RknnWrapper wrapper (model_name, 3);

    while (capture.isOpened())
    {
        cv::Mat img;
        if (capture.read(img) == false)
            break;

        if (!wrapper.EnqueueMat(&img)) {
            break;
        }
        if (!wrapper.DequeueMat(&img)) {
            break;
        }

        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            break;
    }
    return 0;
}
