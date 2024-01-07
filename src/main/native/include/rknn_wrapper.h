#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"

class RknnWrapper
{
private:
    rknnPool<rkYolov5s, cv::Mat, cv::Mat> testPool;

public:
    RknnWrapper(const char *model_name, int threadNum) : testPool(model_name, threadNum)
    {
        if (testPool.init() != 0)
        {
            throw std::runtime_error("rknnPool init fail!\n");
        }
    }

    bool EnqueueMat(cv::Mat *img)
    {
        return (testPool.put(*img) == 0);
    }

    bool DequeueMat(cv::Mat *img) {
        return (testPool.get(*img) == 0);
    }

    ~RknnWrapper()
    {
        // 清空rknn线程池/Clear the thread pool
        while (true)
        {
            cv::Mat img;
            if (testPool.get(img) != 0)
                break;
        }
    }
};
