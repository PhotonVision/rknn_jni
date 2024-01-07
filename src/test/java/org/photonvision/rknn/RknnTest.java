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

import edu.wpi.first.cscore.CvSink;
import edu.wpi.first.cscore.CvSource;
import edu.wpi.first.util.CombinedRuntimeLoader;

public class RknnTest {
    
    private static List<String> getOutputNames(Net net) {
        if (net == null) {
            // barf?
            return List.of();
        }

        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach(
                (item) -> names.add(layersNames.get(item - 1))); // unfold and create R-CNN layers from the
        // loaded YOLO model//
        return names;
    }

    @Test
    public void testBasicBlobs() {
        // I really really hate this. Someone should fix this for me
        // var s = new CvSink("guh");
        try {
            CombinedRuntimeLoader.loadLibraries(RknnTest.class, Core.NATIVE_LIBRARY_NAME);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        System.out.println(Core.getBuildInformation());
        System.out.println(Core.OpenCLApiCallError);

        long start, end;
        start = System.currentTimeMillis();
        var net = Dnn.readNetFromDarknet(
                "/home/matt/Downloads/yolov4-csp-swish.cfg",
                "/home/matt/Downloads/yolov4-csp-swish.weights");
        

        end = System.currentTimeMillis();
        System.out.println("Read (ms): " + (end - start));

        System.out.println("Setting backends");

        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV); 
        net.setPreferableTarget(Dnn.DNN_TARGET_OPENCL); 
        // net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV); 
        // net.setPreferableTarget(Dnn.DNN_TARGET_CPU); 
        // net.setPreferableBackend(Dnn.DNN_BACKEND_CUDA); 
        // net.setPreferableTarget(Dnn.DNN_TARGET_CUDA); 

        Mat img = Imgcodecs.imread("src/test/resources/bus.jpg");
        System.out.println(img);

        start = System.currentTimeMillis();
        var blob = Dnn.blobFromImage(img, 1.0 / 255.0, new Size(640, 640));
        end = System.currentTimeMillis();
        System.out.println("Blob (ms): " + (end - start));

        for (int i = 0; i < 10; i++) {
            start = System.currentTimeMillis();
            net.setInput(blob);
            end = System.currentTimeMillis();
            System.out.println("SetInput (ms): " + (end - start));
            List<Mat> result = new ArrayList<>();
            var outBlobNames = getOutputNames(net);
            System.out.println(outBlobNames);
            start = System.currentTimeMillis();
            net.forward(result, outBlobNames);
            end = System.currentTimeMillis();
            System.out.println("Forward (ms): " + (end - start));
            // System.out.println(result.stream().map(Mat::dump).collect(Collectors.joining(", ")));
        }

        img.release();
        blob.release();
    }
}
