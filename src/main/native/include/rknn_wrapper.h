#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include "yolov8/yolov8.h"
#include <optional>

class RknnWrapper
{
private:
    rkYolov5s yolov5;
    rkYolov8s yolov8;
    int m_numClasses;

public:
    RknnWrapper(const char *model_name, int numClasses, int model_ver)
    { 
        switch (model_ver) {
            case 5:
                yolov5 = rkYolov5s(model_name, numClasses);
                yolov5.init(yolov5.get_pctx(), false);
            break;
            case 8:
                yolov8 = rkYolov8s();
                yolov8.init(model_name);
            break;
        }
    }

    detect_result_group_t forward(cv::Mat &img, DetectionFilterParams params) {
        detect_result_group_t ret;
        int code = yolo.inferOnly(img, &ret, params);
        return ret;
    }

    // Let the rkYolov5s dtor take care of cleanup
    ~RknnWrapper() = default;
};
