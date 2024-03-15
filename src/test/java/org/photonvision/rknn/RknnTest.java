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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.photonvision.rknn.RknnJNI.ModelVersion;

import edu.wpi.first.cscore.CameraServerCvJNI;
import edu.wpi.first.cscore.CameraServerJNI;
import edu.wpi.first.cscore.CvSink;
import edu.wpi.first.cscore.CvSource;
import edu.wpi.first.cscore.UsbCamera;
import edu.wpi.first.util.CombinedRuntimeLoader;

public class RknnTest {
    
    @Test
    public void testBasicBlobs() {

        try {
            CameraServerJNI.Helper.setExtractOnStaticLoad(false);
            CameraServerCvJNI.Helper.setExtractOnStaticLoad(false);
        
            CombinedRuntimeLoader.loadLibraries(RknnTest.class, Core.NATIVE_LIBRARY_NAME, "cscorejni");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        System.out.println(Core.getBuildInformation());
        System.out.println(Core.OpenCLApiCallError);

         
        System.out.println("Loading bus");

        System.out.println("Loading rknn-jni");
        System.load("/home/coolpi/rknn_jni/cmake_build/librknn_jni.so");

        System.out.println("Creating detector on three cores");
        long ptr = RknnJNI.create("/home/coolpi/rknn_jni/note-640-640-yolov5s.rknn", 1, ModelVersion.YOLO_V5.ordinal(), 210);
        
        System.err.println("Grabbing camera");
        var cams = CameraServerJNI.enumerateUsbCameras();
        System.out.println(Arrays.toString(cams));
        var cam = new UsbCamera("RknnTest", cams[0].path);
        CvSink cvSink = new CvSink("opencv_USB Camera 0");
        cvSink.setSource(cam);
        
        System.out.println("Running detector");
        var img = new Mat();
        for (int i = 0; i < 100; i++) {
            cvSink.grabFrame(img);

            var ret = RknnJNI.detect(ptr, img.getNativeObjAddr(), .45, .25);
            System.out.println(Arrays.toString(ret));
        }
        img.release();
        cvSink.close();
        cam.close();

        System.out.println("Killing detector");
        RknnJNI.destroy(ptr);
    }
}
