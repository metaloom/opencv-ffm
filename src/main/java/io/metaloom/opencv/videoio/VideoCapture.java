package io.metaloom.opencv.videoio;

import io.metaloom.opencv.NativeBindings;
import io.metaloom.opencv.core.CvException;
import io.metaloom.opencv.core.Mat;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static io.metaloom.opencv.core.Mat.invoke;
import static io.metaloom.opencv.core.Mat.rethrow;

/**
 * FFM-backed wrapper around native cv::VideoCapture.
 * Implements AutoCloseable for deterministic native resource management.
 */
public class VideoCapture implements AutoCloseable {

    private MemorySegment nativePtr;

    /** Create an uninitialized VideoCapture. */
    public VideoCapture() {
        MemorySegment ptr = invoke(() -> (MemorySegment) NativeBindings.VIDEOCAPTURE_CREATE.invokeExact());
        if (ptr == null || ptr.equals(MemorySegment.NULL)) {
            throw new CvException("Failed to create VideoCapture");
        }
        this.nativePtr = ptr;
    }

    /**
     * Open a video file or stream.
     *
     * @param filename path to a video file, video stream URL, or device identifier
     * @return true if the capture was successfully opened
     */
    public boolean open(String filename) {
        return open(filename, Videoio.CAP_ANY);
    }

    /**
     * Open a video file or stream with a specific API backend.
     *
     * @param filename      path to a video file, video stream URL, or device identifier
     * @param apiPreference preferred capture API backend (see {@link Videoio})
     * @return true if the capture was successfully opened
     */
    public boolean open(String filename, int apiPreference) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment filenameAddr = arena.allocateFrom(filename);
            return (int) NativeBindings.VIDEOCAPTURE_OPEN_FILE.invokeExact(nativePtr, filenameAddr, apiPreference) != 0;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Open a camera device.
     *
     * @param index camera device index (e.g. 0 for default camera)
     * @return true if the capture was successfully opened
     */
    public boolean open(int index) {
        return open(index, Videoio.CAP_ANY);
    }

    /**
     * Open a camera device with a specific API backend.
     *
     * @param index         camera device index
     * @param apiPreference preferred capture API backend
     * @return true if the capture was successfully opened
     */
    public boolean open(int index, int apiPreference) {
        checkReleased();
        return invoke(() -> (int) NativeBindings.VIDEOCAPTURE_OPEN_DEVICE.invokeExact(nativePtr, index, apiPreference)) != 0;
    }

    /**
     * Check if the video capture has been successfully opened.
     */
    public boolean isOpened() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.VIDEOCAPTURE_IS_OPENED.invokeExact(nativePtr)) != 0;
    }

    /**
     * Grabs, decodes and returns the next video frame into the given Mat.
     *
     * @param image Mat to store the frame
     * @return true if a frame was successfully read
     */
    public boolean read(Mat image) {
        checkReleased();
        return invoke(() -> (int) NativeBindings.VIDEOCAPTURE_READ.invokeExact(nativePtr, image.nativePtr())) != 0;
    }

    /**
     * Grabs the next frame from the video source. Use {@link #retrieve(Mat, int)} to decode.
     *
     * @return true if a frame was successfully grabbed
     */
    public boolean grab() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.VIDEOCAPTURE_GRAB.invokeExact(nativePtr)) != 0;
    }

    /**
     * Decodes and returns a grabbed frame.
     *
     * @param image Mat to store the decoded frame
     * @param flag  stream index or channel (usually 0)
     * @return true if a frame was successfully retrieved
     */
    public boolean retrieve(Mat image, int flag) {
        checkReleased();
        return invoke(() -> (int) NativeBindings.VIDEOCAPTURE_RETRIEVE.invokeExact(nativePtr, image.nativePtr(), flag)) != 0;
    }

    public boolean retrieve(Mat image) {
        return retrieve(image, 0);
    }

    /**
     * Set a video capture property.
     *
     * @param propId property identifier (see {@link Videoio} constants)
     * @param value  new property value
     * @return true if the property was set successfully
     */
    public boolean set(int propId, double value) {
        checkReleased();
        return invoke(() -> (int) NativeBindings.VIDEOCAPTURE_SET.invokeExact(nativePtr, propId, value)) != 0;
    }

    /**
     * Get a video capture property.
     *
     * @param propId property identifier (see {@link Videoio} constants)
     * @return property value
     */
    public double get(int propId) {
        checkReleased();
        return invoke(() -> (double) NativeBindings.VIDEOCAPTURE_GET.invokeExact(nativePtr, propId));
    }

    /**
     * Release the video capture resources without deleting the native object.
     */
    public void release() {
        if (nativePtr != null && !nativePtr.equals(MemorySegment.NULL)) {
            try {
                NativeBindings.VIDEOCAPTURE_RELEASE.invokeExact(nativePtr);
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }
    }

    @Override
    public void close() {
        if (nativePtr != null && !nativePtr.equals(MemorySegment.NULL)) {
            try {
                NativeBindings.VIDEOCAPTURE_DELETE.invokeExact(nativePtr);
            } catch (Throwable t) {
                throw rethrow(t);
            }
            nativePtr = MemorySegment.NULL;
        }
    }

    private void checkReleased() {
        if (nativePtr == null || nativePtr.equals(MemorySegment.NULL)) {
            throw new CvException("VideoCapture has been released");
        }
    }
}
