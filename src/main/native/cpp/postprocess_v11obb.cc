#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
#include <limits>
#include <iostream>

#include "yolov11obb/postprocess_v11obb.h"

using std::vector;
using std::min;
using std::max;


// --- Helper functions ---

// Sigmoid function
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Compute softmax for an array of length len
static vector<float> softmax(const float* data, int len) {
    vector<float> out(len);
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < len; i++) {
        if (data[i] > max_val)
            max_val = data[i];
    }
    float sum = 0.f;
    for (int i = 0; i < len; i++) {
        out[i] = std::exp(data[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < len; i++) {
        out[i] /= sum;
    }
    return out;
}

// Rotate an axis-aligned rectangle defined by (x1,y1) and (x2,y2) by angle (in radians).
// Returns four points representing the rotated rectangle in the following order:
// [ rotation(x1,y1), rotation(x1,y2), rotation(x2,y2), rotation(x2,y1) ]
static vector<cv::Point2f> rotate_rectangle(float x1, float y1, float x2, float y2, float angle) {
    float cx = (x1 + x2) / 2.f;
    float cy = (y1 + y2) / 2.f;
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);

    vector<cv::Point2f> pts(4);
    pts[0] = cv::Point2f((x1 - cx) * cosA - (y1 - cy) * sinA + cx,
                         (x1 - cx) * sinA + (y1 - cy) * cosA + cy);
    pts[1] = cv::Point2f((x1 - cx) * cosA - (y2 - cy) * sinA + cx,
                         (x1 - cx) * sinA + (y2 - cy) * cosA + cy);
    pts[2] = cv::Point2f((x2 - cx) * cosA - (y2 - cy) * sinA + cx,
                         (x2 - cx) * sinA + (y2 - cy) * cosA + cy);
    pts[3] = cv::Point2f((x2 - cx) * cosA - (y1 - cy) * sinA + cx,
                         (x2 - cx) * sinA + (y1 - cy) * cosA + cy);
    
    // To mimic the Python order ([pt0, pt3, pt1, pt2]):
    vector<cv::Point2f> rotated;
    rotated.push_back(pts[0]);
    rotated.push_back(pts[3]);
    rotated.push_back(pts[2]);
    rotated.push_back(pts[1]);
    return rotated;
}

// Compute IoU between two rotated boxes given as a polygon (list of 4 cv::Point2f).
static float computeIoU(const vector<cv::Point2f>& poly1, const vector<cv::Point2f>& poly2) {
    // Compute area for each polygon using contourArea
    float area1 = std::fabs(cv::contourArea(poly1));
    float area2 = std::fabs(cv::contourArea(poly2));

    // Compute intersection polygon using OpenCV's convex polygon intersection
    vector<cv::Point2f> interPoly;

    // intersectConvexConvex returns the intersection area.
    float interArea = cv::intersectConvexConvex(poly1, poly2, interPoly, true);
    float unionArea = area1 + area2 - interArea;
    if (unionArea <= 0.f)
        return 0.f;
    
    return interArea / unionArea;
}

// Structure to hold intermediate detection results
struct DetectBox {
    int classId;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float angle; // in radians
    bool suppressed; // flag for NMS
    DetectBox(int cid, float s, float x1, float y1, float x2, float y2, float a)
        : classId(cid), score(s), xmin(x1), ymin(y1), xmax(x2), ymax(y2), angle(a), suppressed(false) {}
};

// This function processes the raw RKNN outputs (assuming three detection branches and one angle branch)
// to produce oriented bounding boxes
int post_process_v11obb(cv::Size modelSize,
                         rknn_output *outputs,
                         BOX_RECT *padding,
                         float conf_threshold,
                         float nms_threshold,
                         detect_result_group_t *od_results,
                         int numClasses,
                         std::vector<rknn_tensor_attr> &output_attrs,
                         bool is_quant,
                         int n_outputs) {
    if(n_outputs < 4) {
        return -1; // error: insufficient outputs
    }
    
    // We'll accumulate detection boxes from all detection branches here.
    vector<DetectBox> detBoxes;

    // --- Angle branch: assume the LAST output is the angle tensor
    const int angle_out_idx = n_outputs - 1;
    const rknn_tensor_attr& angle_attr = output_attrs[angle_out_idx];

    // Data pointers for angle tensor
    const float*  angle_data_f = nullptr;
    const int8_t* angle_data_q = nullptr;
    if (is_quant) {
        angle_data_q = reinterpret_cast<const int8_t*>(outputs[angle_out_idx].buf);
    } else {
        angle_data_f = reinterpret_cast<const float*>(outputs[angle_out_idx].buf);
    }

    // We'll treat the angle tensor as a single flattened buffer.
    // As we iterate detection heads (branches), we advance this offset by H*W.
    size_t angle_offset = 0;

    // --- Process each detection head; assume all but the last are box+class heads
    for (int branch = 0; branch < angle_out_idx; ++branch) {
        const rknn_tensor_attr& attr = output_attrs[branch];

        const int N = (attr.n_dims > 0) ? static_cast<int>(attr.dims[0]) : 1;
        const int C = (attr.n_dims > 1) ? static_cast<int>(attr.dims[1]) : 0;
        const int H = (attr.n_dims > 2) ? static_cast<int>(attr.dims[2]) : 0;
        const int W = (attr.n_dims > 3) ? static_cast<int>(attr.dims[3]) : 0;

        if (N != 1 || C <= 0 || H <= 0 || W <= 0) {
            // skip this head safely
            continue;
        }

        const int spatial_size = H * W;

        // --- Infer layout: channels = loc_channels + numClasses
        const int loc_channels = C - numClasses;
        if (loc_channels < 4 || (loc_channels % 4) != 0) {
            // Not a 4 layout; skip
            continue;
        }

        const int bins_per_side = loc_channels / 4; // 16 normally

        // Confidence starts at channel = loc_channels
        auto conf_channel_of = [&](int cls) { return loc_channels + cls; };

        // Decode stride from feature map scale
        int stride = modelSize.height / H;
        if (stride <= 0) stride = 1;

        // Pointers to detection head data
        const float*  det_data_f = nullptr;
        const int8_t* det_data_q = nullptr;
        if (is_quant) {
            det_data_q = reinterpret_cast<const int8_t*>(outputs[branch].buf);
        } else {
            det_data_f = reinterpret_cast<const float*>(outputs[branch].buf);
        }

        // Iterate all cells × classes
        for (int cls = 0; cls < numClasses; ++cls) {
            const int conf_ch = conf_channel_of(cls);
            // Walk H×W
            for (int idx = 0; idx < spatial_size; ++idx) {
                const int h_idx = idx / W;
                const int w_idx = idx % W;

                // --- confidence
                const int conf_index = conf_ch * spatial_size + idx;
                float conf_val = 0.f;
                if (is_quant) {
                    conf_val = (static_cast<float>(det_data_q[conf_index]) - attr.zp) * attr.scale;
                } else {
                    conf_val = det_data_f[conf_index];
                }
                conf_val = sigmoid(conf_val);
                if (conf_val < conf_threshold) continue;

                // --- DFL decode for 4 coords with bins_per_side
                float coords[4] = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 4; ++i) {
                    // gather logits for this side: channels [i*bins .. i*bins+bins-1] at (h,w)
                    std::vector<float> logits(bins_per_side);
                    for (int b = 0; b < bins_per_side; ++b) {
                        const int ch = i * bins_per_side + b;
                        const int off = ch * spatial_size + idx;
                        float v = 0.f;
                        if (is_quant) {
                            v = (static_cast<float>(det_data_q[off]) - attr.zp) * attr.scale;
                        } else {
                            v = det_data_f[off];
                        }
                        logits[b] = v;
                    }
                    // softmax -> probs
                    std::vector<float> probs = softmax(logits.data(), bins_per_side);
                    // soft-argmax
                    float s = 0.f;
                    for (int b = 0; b < bins_per_side; ++b) s += b * probs[b];
                    coords[i] = s;
                }

                // sums and halves
                const float xywh_add0 = coords[0] + coords[2];
                const float xywh_add1 = coords[1] + coords[3];
                const float xywh_sub0 = (coords[2] - coords[0]) * 0.5f;
                const float xywh_sub1 = (coords[3] - coords[1]) * 0.5f;

                // --- angle from the flattened angle tensor
                const int angle_index = static_cast<int>(angle_offset) + idx; // idx = h*W + w
                float angle_raw = 0.f;
                if (is_quant) {
                    angle_raw = (static_cast<float>(angle_data_q[angle_index]) - angle_attr.zp) * angle_attr.scale;
                } else {
                    angle_raw = angle_data_f[angle_index];
                }
                const float angle_feature = (angle_raw - 0.25f) * 3.14159265358979323846f;

                const float cos_a = std::cos(angle_feature);
                const float sin_a = std::sin(angle_feature);

                const float xy_mul1 = xywh_sub0 * cos_a;
                const float xy_mul2 = xywh_sub1 * sin_a;
                const float xy_mul3 = xywh_sub0 * sin_a;
                const float xy_mul4 = xywh_sub1 * cos_a;

                float center_x = (xy_mul1 - xy_mul2) + w_idx + 0.5f;
                float center_y = (xy_mul3 + xy_mul4) + h_idx + 0.5f;

                center_x *= stride;
                center_y *= stride;

                const float box_w = xywh_add0 * stride;
                const float box_h = xywh_add1 * stride;

                const float xmin = center_x - box_w * 0.5f;
                const float ymin = center_y - box_h * 0.5f;
                const float xmax = center_x + box_w * 0.5f;
                const float ymax = center_y + box_h * 0.5f;

                detBoxes.emplace_back(cls, conf_val, xmin, ymin, xmax, ymax, angle_feature);
            } // cells
        } // classes

        // advance angle offset by this map's spatial size
        angle_offset += static_cast<size_t>(spatial_size);
    } // branch loop

    // --- NMS
    // Sort detections in descending order by confidence.
    std::sort(detBoxes.begin(), detBoxes.end(), [](const DetectBox &a, const DetectBox &b) {
        return a.score > b.score;
    });

    vector<DetectBox> finalDetections;
    for (size_t i = 0; i < detBoxes.size(); i++) {
        if (detBoxes[i].suppressed)
            continue;
        
        // Get rotated polygon for current box
        vector<cv::Point2f> poly1 = rotate_rectangle(detBoxes[i].xmin, detBoxes[i].ymin,
                                                      detBoxes[i].xmax, detBoxes[i].ymax,
                                                      detBoxes[i].angle);
        finalDetections.push_back(detBoxes[i]);

        // Compare with later detections
        for (size_t j = i + 1; j < detBoxes.size(); j++) {
            if (detBoxes[i].classId != detBoxes[j].classId)
                continue;
            if (detBoxes[j].suppressed)
                continue;
            
            vector<cv::Point2f> poly2 = rotate_rectangle(detBoxes[j].xmin, detBoxes[j].ymin,
                                                          detBoxes[j].xmax, detBoxes[j].ymax,
                                                          detBoxes[j].angle);
            
            float iou = computeIoU(poly1, poly2);
            if (iou > nms_threshold) {
                detBoxes[j].suppressed = true;
            }
        }
    }
    
    // --- Fix up OBBs by adjusting the padding
    od_results->results.clear();

    // Precompute
    const float content_w = static_cast<float>(padding->right - padding->left);
    const float content_h = static_cast<float>(padding->bottom - padding->top);
    const float widthScale = content_w / static_cast<float>(modelSize.width);
    const float heightScale = content_h / static_cast<float>(modelSize.height);

    for (size_t i = 0; i < finalDetections.size(); i++) {
        // don't output more detections than the
        // hard limit
        if (i >= OBJ_NUMB_MAX_SIZE_V11OBB) {
            break;
        }

        const DetectBox& d = finalDetections[i];

        float cx = 0.5f * (d.xmin + d.xmax);
        float cy = 0.5f * (d.ymin + d.ymax);
        float w  =        (d.xmax - d.xmin);
        float h  =        (d.ymax - d.ymin);
        float th = d.angle; // radians

        // Remove letterbox offsets then scale
        cx -= padding->left;
        cy -= padding->top;

        cx *= widthScale;
        cy *= heightScale;
        w  *= widthScale;
        h  *= heightScale;

        // Clamp to content dimensions (center within image; keep width/height >= 1 pixel)
        cx = std::max(0.f, std::min(cx, content_w  - 1.f));
        cy = std::max(0.f, std::min(cy, content_h - 1.f));
        w  = std::max(1.f, std::min(w,  content_w));
        h  = std::max(1.f, std::min(h,  content_h));

        detect_result_t result;
        result.id        = d.classId;
        result.obj_conf  = d.score;

        result.obb.cx   = static_cast<int>(std::round(cx));
        result.obb.cy = static_cast<int>(std::round(cy));
        result.obb.width = static_cast<int>(std::round(w));
        result.obb.height = static_cast<int>(std::round(h));

        result.obb.angle = th;  // radians about (center_x, center_y)

        od_results->results.push_back(result);
    }

    od_results->count = static_cast<int>(od_results->results.size());

    return 0;
}