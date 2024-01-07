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

#include "rknn_yolo_wrapper.h"
#include "wpi_jni_common.h"

static JClass detectionResultClass;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  detectionResultClass = JClass(env, "org/photonvision/RknnJNI$RknnResult");

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
  return env->NewObject(detectionResultClass, constructor, result.box.left,
                        result.box.top, result.box.right, result.box.bottom,
                        result.prop, result.cls_id);
}

/*
 * Class:     src_main_java_org_photonvision_rknn_RknnJNI
 * Method:    create
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_src_main_java_org_photonvision_rknn_RknnJNI_create
  (JNIEnv *env, jclass)
{
  return reinterpret_cast<jlong>(new RknnYoloWrapper());
}

/*
 * Class:     src_main_java_org_photonvision_rknn_RknnJNI
 * Method:    detect
 * Signature: (JJIIF)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_src_main_java_org_photonvision_rknn_RknnJNI_detect
  (JNIEnv *env, jclass, jlong detector_, jlong blob, jint x_pad, jint y_pad,
   jfloat scale)
{
  RknnYoloWrapper *yolo = reinterpret_cast<RknnYoloWrapper *>(detector_);

  auto results = yolo->inference(
      reinterpret_cast<uint8_t *>(blob),
      letterbox_t{.x_pad = x_pad, .y_pad = y_pad, .scale = scale});

  if (!results) {
    return {};
  }

  jobjectArray jarr =
      env->NewObjectArray(results.value().count, detectionResultClass, nullptr);

  for (size_t i = 0; i < results.value().count; i++) {
    jobject obj = MakeJObject(env, results.value().results[i]);
    env->SetObjectArrayElement(jarr, i, obj);
  }

  return jarr;
}
