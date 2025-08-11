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

/* DO NOT EDIT THIS std::FILE - it is machine generated */
#include <jni.h>

/* Header for class org_photonvision_rknn_RknnJNI */

#ifndef RKNN_JNI_SRC_MAIN_NATIVE_CPP_RKNN_JNI_H_
#define RKNN_JNI_SRC_MAIN_NATIVE_CPP_RKNN_JNI_H_
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_photonvision_rknn_RknnJNI
 * Method:    create
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_photonvision_rknn_RknnJNI_create(JNIEnv *,
                                                                  jclass,
                                                                  jstring, jint,
                                                                  jint, jint);

JNIEXPORT jint JNICALL Java_org_photonvision_rknn_RknnJNI_setCoreMask(JNIEnv *,
                                                                      jclass,
                                                                      jlong,
                                                                      jint);

/*
 * Class:     org_photonvision_rknn_RknnJNI
 * Method:    destroy
 * Signature: (J)J
 */
JNIEXPORT void JNICALL Java_org_photonvision_rknn_RknnJNI_destroy(JNIEnv *,
                                                                  jclass,
                                                                  jlong);

/*
 * Class:     org_photonvision_rknn_RknnJNI
 * Method:    detect
 * Signature: (JJIIF)[Lorg/photonvision/rknn/RknnJNI/RknnResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_photonvision_rknn_RknnJNI_detect(
    JNIEnv *, jclass, jlong, jlong, jdouble, jdouble);


/*
 * Class:     org_photonvision_rknn_RknnJNI
 * Method:    isQuantized
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_org_photonvision_rknn_RknnJNI_isQuantized
(JNIEnv *, jclass, jlong);


#ifdef __cplusplus
} // extern "C"
#endif
#endif // RKNN_JNI_SRC_MAIN_NATIVE_CPP_RKNN_JNI_H_
