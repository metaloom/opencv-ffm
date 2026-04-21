package io.metaloom.opencv.objdetect;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

import io.metaloom.opencv.NativeBindings;
import io.metaloom.opencv.core.Mat;
import io.metaloom.opencv.core.Rect;

/**
 * FFM-based wrapper for cv::CascadeClassifier.
 */
public class CascadeClassifier {

    private static final int MAX_DETECTIONS = 1024;

    private MemorySegment handle;

    public CascadeClassifier() {
        try {
            handle = (MemorySegment) NativeBindings.CASCADE_CREATE.invoke();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create CascadeClassifier", e);
        }
    }

    /**
     * Load a classifier from a file.
     *
     * @param path path to the XML cascade file
     * @return true if loaded successfully
     */
    public boolean load(String path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment pathSeg = arena.allocateFrom(path);
            int result = (int) NativeBindings.CASCADE_LOAD.invoke(handle, pathSeg);
            return result != 0;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to load cascade: " + path, e);
        }
    }

    /**
     * Detect objects with default parameters.
     *
     * @param mat input image
     * @return list of detected rectangles
     */
    public List<Rect> detectMultiScale(Mat mat) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outRects = arena.allocate(ValueLayout.JAVA_INT, (long) MAX_DETECTIONS * 4);
            int count = (int) NativeBindings.CASCADE_DETECT.invoke(
                    handle, mat.nativePtr(), outRects, MAX_DETECTIONS);
            return parseRects(outRects, count);
        } catch (Throwable e) {
            throw new RuntimeException("detectMultiScale failed", e);
        }
    }

    /**
     * Detect objects with custom parameters.
     *
     * @param mat          input image
     * @param scaleFactor  parameter specifying how much the image size is reduced
     * @param minNeighbors parameter specifying how many neighbors each candidate rectangle should have
     * @param flags        deprecated, use 0 or CASCADE_SCALE_IMAGE (4)
     * @param minSize      minimum possible object size
     * @return list of detected rectangles
     */
    public List<Rect> detectMultiScale(Mat mat, double scaleFactor, int minNeighbors, int flags,
            int minSizeWidth, int minSizeHeight) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outRects = arena.allocate(ValueLayout.JAVA_INT, (long) MAX_DETECTIONS * 4);
            int count = (int) NativeBindings.CASCADE_DETECT_PARAMS.invoke(
                    handle, mat.nativePtr(),
                    scaleFactor, minNeighbors, flags,
                    minSizeWidth, minSizeHeight,
                    outRects, MAX_DETECTIONS);
            return parseRects(outRects, count);
        } catch (Throwable e) {
            throw new RuntimeException("detectMultiScale failed", e);
        }
    }

    private List<Rect> parseRects(MemorySegment outRects, int count) {
        List<Rect> result = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int x = outRects.getAtIndex(ValueLayout.JAVA_INT, i * 4L);
            int y = outRects.getAtIndex(ValueLayout.JAVA_INT, i * 4L + 1);
            int w = outRects.getAtIndex(ValueLayout.JAVA_INT, i * 4L + 2);
            int h = outRects.getAtIndex(ValueLayout.JAVA_INT, i * 4L + 3);
            result.add(new Rect(x, y, w, h));
        }
        return result;
    }

    public void release() {
        if (handle != null) {
            try {
                NativeBindings.CASCADE_DELETE.invoke(handle);
            } catch (Throwable e) {
                // ignore
            }
            handle = null;
        }
    }
}
