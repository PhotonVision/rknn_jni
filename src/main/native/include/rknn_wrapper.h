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
    int m_numClasses;

public:
    RknnWrapper(const char *model_name, int numClasses, int model_ver) : yolo(model_name, numClasses)
    { 
        yolo.init(yolo.get_pctx(), false);
    }

    detect_result_group_t forward(cv::Mat &img, DetectionFilterParams params) {
        detect_result_group_t ret;
        int code = yolo.inferOnly(img, &ret, params);
        return ret;
    }

    // Let the rkYolov5s dtor take care of cleanup
    ~RknnWrapper() = default;
};
