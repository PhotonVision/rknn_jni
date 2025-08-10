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

package org.photonvision.rknn;

import edu.wpi.first.util.CombinedRuntimeLoader;
import java.io.IOException;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.photonvision.rknn.RknnJNI.ModelVersion;

import static org.junit.jupiter.api.Assertions.*;

public class RknnTest {
    @Test
    public void testBasicBlobs() {
        try {
            CombinedRuntimeLoader.loadLibraries(RknnTest.class, Core.NATIVE_LIBRARY_NAME);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        System.out.println(Core.getBuildInformation());
        System.out.println(Core.OpenCLApiCallError);

        System.out.println("Loading bus");
        Mat img = Imgcodecs.imread("silly_notes.png");
        Mat img2 = Imgcodecs.imread("silly_notes2.png");

        System.out.println("Loading rknn-jni");
        System.load("/home/coolpi/rknn_jni/cmake_build/librknn_jni.so");

        System.out.println("Creating detector on three cores");
        long ptr =
                RknnJNI.create(
                        "/home/coolpi/rknn_jni/note-640-640-yolov5s.rknn",
                        1,
                        ModelVersion.YOLO_V5.ordinal(),
                        210);

        System.out.println("Running detector");
        var ret = RknnJNI.detect(ptr, img.getNativeObjAddr(), .45, .25);
        System.out.println(Arrays.toString(ret));

        System.out.println("Changing detector to run on core 0");
        System.out.println("return code: " + RknnJNI.setCoreMask(ptr, 0));

        System.out.println("Running detector again");
        ret = RknnJNI.detect(ptr, img2.getNativeObjAddr(), .45, .25);
        System.out.println(Arrays.toString(ret));

        System.out.println("Killing detector");
        RknnJNI.destroy(ptr);
    }

    private boolean checkQuant(String path, int numClasses, ModelVersion modelVer) {
        long ptr = RknnJNI.create(path, numClasses, modelVer.ordinal(), 210);

        if (ptr <= 0) {
            throw new RuntimeException("Failed to create model");
        }

        boolean isQuantized = RknnJNI.isQuantized(ptr);

        RknnJNI.destroy(ptr);

        return isQuantized;
    }

    @Test
    public void testQuantizationCheck() {
        assertTrue(checkQuant("/home/coolpi/rknn_jni/note-640-640-yolov5s.rknn", 1, ModelVersion.YOLO_V5));
        assertFalse(checkQuant("/home/coolpi/rknn_jni/coralAlgaeModelNoQuant-640-640-yolov8s.rknn", 2, ModelVersion.YOLO_V8));
    }
}
