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

/* Header for class src_main_java_org_photonvision_rknn_RknnJNI */

#ifndef RKNN_JAVA_RKNN_JNI_H_
#define RKNN_JAVA_RKNN_JNI_H_
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     src_main_java_org_photonvision_rknn_RknnJNI
 * Method:    detect
 * Signature: (JIIF)[Lsrc/main/java/org/photonvision/rknn/RknnJNI/RknnResult;
 */
JNIEXPORT jobjectArray JNICALL
Java_src_main_java_org_photonvision_rknn_RknnJNI_detect(JNIEnv *, jclass, jlong,
                                                        jlong, jint, jint,
                                                        jfloat);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // RKNN_JAVA_RKNN_JNI_H_
