package src.main.java.org.photonvision.rknn;

public class RknnJNI {
    public static class RknnResult {
        public RknnResult(
            int left, int top, int right, int bottom, float conf, int class_id
        ) {}
    }

    public static native RknnResult[] detect(
        long detectorPtr, long blobNCHWPtr, 
        // letterbox config for rescaling output
        int x_pad, int y_pad, float scale
    );
}
