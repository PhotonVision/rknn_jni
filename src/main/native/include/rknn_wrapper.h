#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include <optional>

class RknnWrapper
{
private:
    rkYolov5s yolo;

public:
    RknnWrapper(const char *model_name) 
        : yolo(model_name) 
    { }

    detect_result_group_t forward(cv::Mat &img) {
        detect_result_group_t ret;
        int code = yolo.inferOnly(img, &ret);
        return ret;
    }

    // Let the rkYolov5s dtor take care of cleanup
    ~RknnWrapper() = default;
};