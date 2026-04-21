package io.metaloom.opencv.video;

import java.lang.foreign.MemorySegment;

import io.metaloom.opencv.NativeBindings;
import io.metaloom.opencv.core.CvException;
import io.metaloom.opencv.core.Mat;
import io.metaloom.opencv.core.Size;

import static io.metaloom.opencv.core.Mat.invoke;

public class SparsePyrLKOpticalFlow {

    private static final int TERM_CRITERIA_COUNT = 1;
    private static final int TERM_CRITERIA_EPS = 2;

    private final Size winSize;
    private final int maxLevel;
    private final int criteriaType;
    private final int criteriaMaxCount;
    private final double criteriaEpsilon;
    private final int flags;
    private final double minEigThreshold;

    private SparsePyrLKOpticalFlow(Size winSize, int maxLevel, int criteriaType,
            int criteriaMaxCount, double criteriaEpsilon, int flags, double minEigThreshold) {
        this.winSize = winSize;
        this.maxLevel = maxLevel;
        this.criteriaType = criteriaType;
        this.criteriaMaxCount = criteriaMaxCount;
        this.criteriaEpsilon = criteriaEpsilon;
        this.flags = flags;
        this.minEigThreshold = minEigThreshold;
    }

    public static SparsePyrLKOpticalFlow create() {
        return new SparsePyrLKOpticalFlow(
                new Size(21, 21),
                3,
                TERM_CRITERIA_COUNT | TERM_CRITERIA_EPS,
                30,
                0.01,
                0,
                1e-4);
    }

    public static SparsePyrLKOpticalFlow create(Size winSize, int maxLevel, int criteriaType,
            int criteriaMaxCount, double criteriaEpsilon, int flags, double minEigThreshold) {
        return new SparsePyrLKOpticalFlow(winSize, maxLevel, criteriaType,
                criteriaMaxCount, criteriaEpsilon, flags, minEigThreshold);
    }

    public void calc(Mat prevImg, Mat nextImg, Mat prevPts, Mat nextPts, Mat status, Mat err) {
        int res = invoke(() -> (int) NativeBindings.VIDEO_CALC_OPTICAL_FLOW_PYR_LK.invokeExact(
                prevImg.nativePtr(),
                nextImg.nativePtr(),
                prevPts.nativePtr(),
                nextPts.nativePtr(),
                status.nativePtr(),
                err.nativePtr(),
                winSize.width,
                winSize.height,
                maxLevel,
                criteriaType,
                criteriaMaxCount,
                criteriaEpsilon,
                flags,
                minEigThreshold));
        checkError(res);
    }

    private static void checkError(int result) {
        if (result < 0) {
            String msg = getLastError();
            throw new CvException(msg != null ? msg : "SparsePyrLKOpticalFlow operation failed with code: " + result);
        }
    }

    private static String getLastError() {
        try {
            MemorySegment errPtr = (MemorySegment) NativeBindings.GET_LAST_ERROR.invokeExact();
            if (!errPtr.equals(MemorySegment.NULL)) {
                return errPtr.reinterpret(1024).getString(0);
            }
        } catch (Throwable ignored) {
        }
        return null;
    }
}
