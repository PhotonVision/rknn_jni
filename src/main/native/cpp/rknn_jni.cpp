/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "rknn_jni.h"

#include <cstdio>

#include "postprocess.h"
#include "rkYolov5s.hpp"
#include "rknn_wrapper.h"
#include "wpi_jni_common.h"

static JClass detectionResultClass;

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  detectionResultClass =
      JClass(env, "org/photonvision/rknn/RknnJNI$RknnResult");

  if (!detectionResultClass) {
    std::printf("Couldn't find class!");
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

static jobject MakeJObject(JNIEnv *env, const detect_result_t &result) {
  jmethodID constructor =
      env->GetMethodID(detectionResultClass, "<init>", "(IIIIFI)V");

  // Actually call the constructor
  return env->NewObject(detectionResultClass, constructor, result.box.left,
                        result.box.top, result.box.right, result.box.bottom,
                        result.obj_conf, result.id);
}

/*
 * Class:     org_photonvision_rknn_RknnJNI
 * Method:    create
 * Signature: (Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL
Java_org_photonvision_rknn_RknnJNI_create
  (JNIEnv *env, jclass, jstring javaString, jint numClasses, jint modelVer)
{
  const char *nativeString = env->GetStringUTFChars(javaString, 0);
  std::printf("Creating for %s\n", nativeString);

  auto ret = new RknnWrapper(nativeString, numClasses, reinterpret_cast<ModelVersion>(modelVer));
  env->ReleaseStringUTFChars(javaString, nativeString);
  return reinterpret_cast<jlong>(ret);
}

/*
 * Class:     org_photonvision_rknn_RknnJNI
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_photonvision_rknn_RknnJNI_destroy
  (JNIEnv *env, jclass, jlong ptr)
{
  delete reinterpret_cast<RknnWrapper *>(ptr);
}

/*
 * Class:     org_photonvision_rknn_RknnJNI
 * Method:    detect
 * Signature: (JJDDI)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_photonvision_rknn_RknnJNI_detect
  (JNIEnv *env, jclass, jlong detector_, jlong input_cvmat_ptr,
   jdouble nms_thresh, jdouble box_thresh)
{
  RknnWrapper *yolo = reinterpret_cast<RknnWrapper *>(detector_);
  cv::Mat *input_img = reinterpret_cast<cv::Mat *>(input_cvmat_ptr);

  DetectionFilterParams params{
      .nms_thresh = nms_thresh,
      .box_thresh = box_thresh,
  };
  auto results = yolo->forward(*input_img, params);

  if (results.count < 1) {
    return nullptr;
  }

  jobjectArray jarr =
      env->NewObjectArray(results.count, detectionResultClass, nullptr);

  for (size_t i = 0; i < results.count; i++) {
    jobject obj = MakeJObject(env, results.results[i]);
    env->SetObjectArrayElement(jarr, i, obj);
  }

  return jarr;
}

} // extern "C"
