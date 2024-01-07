#include "rknn_jni.h"
#include "rknn_yolo_wrapper.h"
#include <cstdio>
#include "wpi_jni_common.h"

static JClass detectionResultClass;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv((void **)(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  detectionResultClass =
      JClass(env, "org/photonvision/RknnJNI$RknnResult");

  if (!detectionResultClass) {
    std::printf("Couldn't find class!");
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

static jobject MakeJObject(JNIEnv *env, const object_detect_result &result) {
  jmethodID constructor =
      env->GetMethodID(detectionResultClass, "<init>", "(IIIIFI)V");

  // Actually call the constructor
  return env->NewObject(detectionResultClass, constructor, 
    result.box.left, 
    result.box.top, 
    result.box.right, 
    result.box.bottom, 
    result.prop, 
    result.cls_id
  );
}

JNIEXPORT jlong JNICALL Java_src_main_java_org_photonvision_rknn_RknnJNI_create
  (JNIEnv *env, jclass) {
    return reinterpret_cast<jlong>(new RknnYoloWrapper());
  }

/*
 * Class:     src_main_java_org_photonvision_rknn_RknnJNI
 * Method:    detect
 * Signature: (JIIF)[Lsrc/main/java/org/photonvision/rknn/RknnJNI/RknnResult;
 */
JNIEXPORT jobjectArray JNICALL Java_src_main_java_org_photonvision_rknn_RknnJNI_detect
  (JNIEnv *env, jclass, jlong detector_, jlong blob, jint x_pad, jint y_pad, jfloat scale) {
    RknnYoloWrapper *yolo = reinterpret_cast<RknnYoloWrapper*>(detector_);

    auto results = yolo->inference(reinterpret_cast<uint8_t*>(blob), letterbox_t {
      .x_pad = x_pad,
      .y_pad = y_pad,
      .scale = scale
    });

    if (!results) {
      return {};
    }

    jobjectArray jarr = env->NewObjectArray(results.value().count, detectionResultClass, nullptr);

    for (size_t i = 0; i < results.value().count; i++) {
      jobject obj = MakeJObject(env, results.value().results[i]);
      env->SetObjectArrayElement(jarr, i, obj);
    }

    return jarr;
}
