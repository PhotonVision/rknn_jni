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

public class RknnJNI {
    public static class RknnResult {
        public RknnResult(
            int left, int top, int right, int bottom, float conf, int class_id
        ) {
            this.left = left;
            this.right = right;
            this.bottom = bottom;
            this.top = top;
            this.conf = conf;
            this.class_id = class_id;

        }
        public String toString() {
            return "Left: " + left + "\nRight: " + right + "\nBottom: " + bottom + "\nTop: " + top + "\nConf: " + conf + "\nClass Id: " + class_id;
        }
        public final int left;
        public final int right;
        public final int bottom;
        public final int top;
        public final float conf;
        public final int class_id;
    }

    public static native long create(String modelPath);
    public static native long destroy(long ptr);

    public static native RknnResult[] detect(
        long detectorPtr, long imagePtr
    );
}
