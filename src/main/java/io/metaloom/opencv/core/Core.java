package io.metaloom.opencv.core;

import io.metaloom.opencv.NativeBindings;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import static io.metaloom.opencv.core.Mat.invoke;
import static io.metaloom.opencv.core.Mat.rethrow;
import static java.lang.foreign.ValueLayout.*;

public class Core {

    // ========================================================================
    // Border types
    // ========================================================================
    public static final int BORDER_CONSTANT    = 0;
    public static final int BORDER_REPLICATE   = 1;
    public static final int BORDER_REFLECT     = 2;
    public static final int BORDER_WRAP        = 3;
    public static final int BORDER_REFLECT_101 = 4;
    public static final int BORDER_TRANSPARENT = 5;
    public static final int BORDER_REFLECT101  = BORDER_REFLECT_101;
    public static final int BORDER_DEFAULT     = BORDER_REFLECT_101;
    public static final int BORDER_ISOLATED    = 16;

    // ========================================================================
    // Comparison types
    // ========================================================================
    public static final int CMP_EQ = 0;
    public static final int CMP_GT = 1;
    public static final int CMP_GE = 2;
    public static final int CMP_LT = 3;
    public static final int CMP_LE = 4;
    public static final int CMP_NE = 5;

    // ========================================================================
    // Norm types
    // ========================================================================
    public static final int NORM_INF      = 1;
    public static final int NORM_L1       = 2;
    public static final int NORM_L2       = 4;
    public static final int NORM_L2SQR    = 5;
    public static final int NORM_HAMMING  = 6;
    public static final int NORM_HAMMING2 = 7;
    public static final int NORM_RELATIVE = 8;
    public static final int NORM_MINMAX   = 32;

    // ========================================================================
    // Decomposition types
    // ========================================================================
    public static final int DECOMP_LU       = 0;
    public static final int DECOMP_SVD      = 1;
    public static final int DECOMP_EIG      = 2;
    public static final int DECOMP_CHOLESKY = 3;
    public static final int DECOMP_QR       = 4;
    public static final int DECOMP_NORMAL   = 16;

    // ========================================================================
    // DFT flags
    // ========================================================================
    public static final int DFT_INVERSE        = 1;
    public static final int DFT_SCALE          = 2;
    public static final int DFT_ROWS           = 4;
    public static final int DFT_COMPLEX_OUTPUT = 16;
    public static final int DFT_REAL_OUTPUT    = 32;
    public static final int DFT_COMPLEX_INPUT  = 64;
    public static final int DCT_INVERSE        = DFT_INVERSE;
    public static final int DCT_ROWS           = DFT_ROWS;

    // ========================================================================
    // GEMM flags
    // ========================================================================
    public static final int GEMM_1_T = 1;
    public static final int GEMM_2_T = 2;
    public static final int GEMM_3_T = 4;

    // ========================================================================
    // Rotate flags
    // ========================================================================
    public static final int ROTATE_90_CLOCKWISE         = 0;
    public static final int ROTATE_180                  = 1;
    public static final int ROTATE_90_COUNTERCLOCKWISE  = 2;

    // ========================================================================
    // Sort flags
    // ========================================================================
    public static final int SORT_EVERY_ROW    = 0;
    public static final int SORT_EVERY_COLUMN = 1;
    public static final int SORT_ASCENDING    = 0;
    public static final int SORT_DESCENDING   = 16;

    // ========================================================================
    // Reduce types
    // ========================================================================
    public static final int REDUCE_SUM = 0;
    public static final int REDUCE_AVG = 1;
    public static final int REDUCE_MAX = 2;
    public static final int REDUCE_MIN = 3;

    private Core() {
    }

    // ========================================================================
    // Arithmetic
    // ========================================================================

    public static void add(Mat src1, Mat src2, Mat dst, Mat mask, int dtype) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        int res = invoke(() -> (int) NativeBindings.CORE_ADD.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr(), maskPtr, dtype));
        checkError(res);
    }

    public static void add(Mat src1, Mat src2, Mat dst) {
        add(src1, src2, dst, null, -1);
    }

    public static void subtract(Mat src1, Mat src2, Mat dst, Mat mask, int dtype) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        int res = invoke(() -> (int) NativeBindings.CORE_SUBTRACT.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr(), maskPtr, dtype));
        checkError(res);
    }

    public static void subtract(Mat src1, Mat src2, Mat dst) {
        subtract(src1, src2, dst, null, -1);
    }

    public static void multiply(Mat src1, Mat src2, Mat dst, double scale, int dtype) {
        int res = invoke(() -> (int) NativeBindings.CORE_MULTIPLY.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr(), scale, dtype));
        checkError(res);
    }

    public static void multiply(Mat src1, Mat src2, Mat dst) {
        multiply(src1, src2, dst, 1.0, -1);
    }

    public static void divide(Mat src1, Mat src2, Mat dst, double scale, int dtype) {
        int res = invoke(() -> (int) NativeBindings.CORE_DIVIDE.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr(), scale, dtype));
        checkError(res);
    }

    public static void divide(Mat src1, Mat src2, Mat dst) {
        divide(src1, src2, dst, 1.0, -1);
    }

    public static void addWeighted(Mat src1, double alpha, Mat src2, double beta, double gamma, Mat dst, int dtype) {
        int res = invoke(() -> (int) NativeBindings.CORE_ADD_WEIGHTED.invokeExact(
                src1.nativePtr(), alpha, src2.nativePtr(), beta, gamma, dst.nativePtr(), dtype));
        checkError(res);
    }

    public static void addWeighted(Mat src1, double alpha, Mat src2, double beta, double gamma, Mat dst) {
        addWeighted(src1, alpha, src2, beta, gamma, dst, -1);
    }

    public static void scaleAdd(Mat src1, double alpha, Mat src2, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_SCALE_ADD.invokeExact(
                src1.nativePtr(), alpha, src2.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    public static void absdiff(Mat src1, Mat src2, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_ABS_DIFF.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    public static void sqrt(Mat src, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_SQRT.invokeExact(
                src.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    // ========================================================================
    // Bitwise
    // ========================================================================

    public static void bitwise_and(Mat src1, Mat src2, Mat dst, Mat mask) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        int res = invoke(() -> (int) NativeBindings.CORE_BITWISE_AND.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr(), maskPtr));
        checkError(res);
    }

    public static void bitwise_and(Mat src1, Mat src2, Mat dst) {
        bitwise_and(src1, src2, dst, null);
    }

    public static void bitwise_or(Mat src1, Mat src2, Mat dst, Mat mask) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        int res = invoke(() -> (int) NativeBindings.CORE_BITWISE_OR.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr(), maskPtr));
        checkError(res);
    }

    public static void bitwise_or(Mat src1, Mat src2, Mat dst) {
        bitwise_or(src1, src2, dst, null);
    }

    public static void bitwise_xor(Mat src1, Mat src2, Mat dst, Mat mask) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        int res = invoke(() -> (int) NativeBindings.CORE_BITWISE_XOR.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr(), maskPtr));
        checkError(res);
    }

    public static void bitwise_xor(Mat src1, Mat src2, Mat dst) {
        bitwise_xor(src1, src2, dst, null);
    }

    public static void bitwise_not(Mat src, Mat dst, Mat mask) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        int res = invoke(() -> (int) NativeBindings.CORE_BITWISE_NOT.invokeExact(
                src.nativePtr(), dst.nativePtr(), maskPtr));
        checkError(res);
    }

    public static void bitwise_not(Mat src, Mat dst) {
        bitwise_not(src, dst, null);
    }

    // ========================================================================
    // Comparison
    // ========================================================================

    public static void compare(Mat src1, Mat src2, Mat dst, int cmpop) {
        int res = invoke(() -> (int) NativeBindings.CORE_COMPARE.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr(), cmpop));
        checkError(res);
    }

    public static void min(Mat src1, Mat src2, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_MIN.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    public static void max(Mat src1, Mat src2, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_MAX.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    public static void inRange(Mat src, Mat lowerb, Mat upperb, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_IN_RANGE.invokeExact(
                src.nativePtr(), lowerb.nativePtr(), upperb.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    public static double norm(Mat src, int normType, Mat mask) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        return invoke(() -> (double) NativeBindings.CORE_NORM.invokeExact(src.nativePtr(), normType, maskPtr));
    }

    public static double norm(Mat src, int normType) {
        return norm(src, normType, null);
    }

    public static double norm(Mat src) {
        return norm(src, NORM_L2, null);
    }

    public static double norm(Mat src1, Mat src2, int normType, Mat mask) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        return invoke(() -> (double) NativeBindings.CORE_NORM_DIFF.invokeExact(
                src1.nativePtr(), src2.nativePtr(), normType, maskPtr));
    }

    public static double norm(Mat src1, Mat src2, int normType) {
        return norm(src1, src2, normType, null);
    }

    public static double norm(Mat src1, Mat src2) {
        return norm(src1, src2, NORM_L2, null);
    }

    public static Scalar mean(Mat src, Mat mask) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(JAVA_DOUBLE, 4);
            MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
            int res = (int) NativeBindings.CORE_MEAN.invokeExact(src.nativePtr(), maskPtr, buf);
            checkError(res);
            double[] vals = new double[4];
            MemorySegment.copy(buf, JAVA_DOUBLE, 0, vals, 0, 4);
            return new Scalar(vals);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static Scalar mean(Mat src) {
        return mean(src, null);
    }

    public static void meanStdDev(Mat src, MatOfDouble mean, MatOfDouble stddev, Mat mask) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment meanBuf = arena.allocate(JAVA_DOUBLE, 4);
            MemorySegment stdBuf = arena.allocate(JAVA_DOUBLE, 4);
            MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
            int res = (int) NativeBindings.CORE_MEAN_STD_DEV.invokeExact(
                    src.nativePtr(), meanBuf, stdBuf, maskPtr);
            checkError(res);
            double[] meanVals = new double[4];
            double[] stdVals = new double[4];
            MemorySegment.copy(meanBuf, JAVA_DOUBLE, 0, meanVals, 0, 4);
            MemorySegment.copy(stdBuf, JAVA_DOUBLE, 0, stdVals, 0, 4);
            mean.fromArray(meanVals);
            stddev.fromArray(stdVals);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static void meanStdDev(Mat src, MatOfDouble mean, MatOfDouble stddev) {
        meanStdDev(src, mean, stddev, null);
    }

    public static MinMaxLocResult minMaxLoc(Mat src, Mat mask) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment minVal = arena.allocate(JAVA_DOUBLE);
            MemorySegment maxVal = arena.allocate(JAVA_DOUBLE);
            MemorySegment minIdx = arena.allocate(JAVA_INT, 2);
            MemorySegment maxIdx = arena.allocate(JAVA_INT, 2);
            MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
            int res = (int) NativeBindings.CORE_MIN_MAX_LOC.invokeExact(
                    src.nativePtr(), minVal, maxVal, minIdx, maxIdx, maskPtr);
            checkError(res);
            MinMaxLocResult result = new MinMaxLocResult();
            result.minVal = minVal.get(JAVA_DOUBLE, 0);
            result.maxVal = maxVal.get(JAVA_DOUBLE, 0);
            result.minLoc = new Point(minIdx.get(JAVA_INT, 0), minIdx.get(JAVA_INT, 4));
            result.maxLoc = new Point(maxIdx.get(JAVA_INT, 0), maxIdx.get(JAVA_INT, 4));
            return result;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static MinMaxLocResult minMaxLoc(Mat src) {
        return minMaxLoc(src, null);
    }

    public static int countNonZero(Mat src) {
        return invoke(() -> (int) NativeBindings.CORE_COUNT_NON_ZERO.invokeExact(src.nativePtr()));
    }

    public static Scalar sumElems(Mat src) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(JAVA_DOUBLE, 4);
            int res = (int) NativeBindings.CORE_SUM.invokeExact(src.nativePtr(), buf);
            checkError(res);
            double[] vals = new double[4];
            MemorySegment.copy(buf, JAVA_DOUBLE, 0, vals, 0, 4);
            return new Scalar(vals);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ========================================================================
    // Array operations
    // ========================================================================

    public static void normalize(Mat src, Mat dst, double alpha, double beta, int norm_type, int dtype, Mat mask) {
        MemorySegment maskPtr = mask != null ? mask.nativePtr() : MemorySegment.NULL;
        int res = invoke(() -> (int) NativeBindings.CORE_NORMALIZE.invokeExact(
                src.nativePtr(), dst.nativePtr(), alpha, beta, norm_type, dtype, maskPtr));
        checkError(res);
    }

    public static void normalize(Mat src, Mat dst, double alpha, double beta, int norm_type) {
        normalize(src, dst, alpha, beta, norm_type, -1, null);
    }

    public static void normalize(Mat src, Mat dst) {
        normalize(src, dst, 1, 0, NORM_L2, -1, null);
    }

    public static void flip(Mat src, Mat dst, int flipCode) {
        int res = invoke(() -> (int) NativeBindings.CORE_FLIP.invokeExact(
                src.nativePtr(), dst.nativePtr(), flipCode));
        checkError(res);
    }

    public static void rotate(Mat src, Mat dst, int rotateCode) {
        int res = invoke(() -> (int) NativeBindings.CORE_ROTATE.invokeExact(
                src.nativePtr(), dst.nativePtr(), rotateCode));
        checkError(res);
    }

    public static void transpose(Mat src, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_TRANSPOSE.invokeExact(
                src.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    public static void convertScaleAbs(Mat src, Mat dst, double alpha, double beta) {
        int res = invoke(() -> (int) NativeBindings.CORE_CONVERT_SCALE_ABS.invokeExact(
                src.nativePtr(), dst.nativePtr(), alpha, beta));
        checkError(res);
    }

    public static void convertScaleAbs(Mat src, Mat dst) {
        convertScaleAbs(src, dst, 1.0, 0.0);
    }

    public static void LUT(Mat src, Mat lut, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_LUT.invokeExact(
                src.nativePtr(), lut.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    public static void copyMakeBorder(Mat src, Mat dst, int top, int bottom, int left, int right,
                                       int borderType, Scalar value) {
        double v0 = value != null ? value.val[0] : 0;
        double v1 = value != null ? value.val[1] : 0;
        double v2 = value != null ? value.val[2] : 0;
        double v3 = value != null ? value.val[3] : 0;
        int res = invoke(() -> (int) NativeBindings.CORE_COPY_MAKE_BORDER.invokeExact(
                src.nativePtr(), dst.nativePtr(), top, bottom, left, right, borderType, v0, v1, v2, v3));
        checkError(res);
    }

    public static void copyMakeBorder(Mat src, Mat dst, int top, int bottom, int left, int right, int borderType) {
        copyMakeBorder(src, dst, top, bottom, left, right, borderType, new Scalar(0));
    }

    public static void hconcat(Mat src1, Mat src2, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_HCONCAT.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    public static void hconcat(java.util.List<Mat> src, Mat dst) {
        if (src == null || src.isEmpty()) {
            throw new CvException("hconcat: input list is empty");
        }
        if (src.size() == 1) {
            src.get(0).copyTo(dst);
            return;
        }
        Mat acc = src.get(0).clone();
        for (int i = 1; i < src.size(); i++) {
            Mat tmp = new Mat();
            hconcat(acc, src.get(i), tmp);
            acc.close();
            acc = tmp;
        }
        acc.copyTo(dst);
        acc.close();
    }

    public static void vconcat(Mat src1, Mat src2, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.CORE_VCONCAT.invokeExact(
                src1.nativePtr(), src2.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    public static void vconcat(java.util.List<Mat> src, Mat dst) {
        if (src == null || src.isEmpty()) {
            throw new CvException("vconcat: input list is empty");
        }
        if (src.size() == 1) {
            src.get(0).copyTo(dst);
            return;
        }
        Mat acc = src.get(0).clone();
        for (int i = 1; i < src.size(); i++) {
            Mat tmp = new Mat();
            vconcat(acc, src.get(i), tmp);
            acc.close();
            acc = tmp;
        }
        acc.copyTo(dst);
        acc.close();
    }

    public static void gemm(Mat src1, Mat src2, double alpha, Mat src3, double beta, Mat dst, int flags) {
        int res = invoke(() -> (int) NativeBindings.CORE_GEMM.invokeExact(
                src1.nativePtr(), src2.nativePtr(), alpha, src3.nativePtr(), beta, dst.nativePtr(), flags));
        checkError(res);
    }

    public static void gemm(Mat src1, Mat src2, double alpha, Mat src3, double beta, Mat dst) {
        gemm(src1, src2, alpha, src3, beta, dst, 0);
    }

    public static void dft(Mat src, Mat dst, int flags, int nonzeroRows) {
        int res = invoke(() -> (int) NativeBindings.CORE_DFT.invokeExact(
                src.nativePtr(), dst.nativePtr(), flags, nonzeroRows));
        checkError(res);
    }

    public static void dft(Mat src, Mat dst) {
        dft(src, dst, 0, 0);
    }

    public static void idft(Mat src, Mat dst, int flags, int nonzeroRows) {
        int res = invoke(() -> (int) NativeBindings.CORE_IDFT.invokeExact(
                src.nativePtr(), dst.nativePtr(), flags, nonzeroRows));
        checkError(res);
    }

    public static void idft(Mat src, Mat dst) {
        idft(src, dst, 0, 0);
    }

    // ========================================================================
    // Merge / Split
    // ========================================================================

    public static void merge(List<Mat> mv, Mat dst) {
        // Build a native array of Mat pointers and use hconcat/vconcat pattern
        // For merge, we call the C wrapper which takes an array
        // Since our C wrapper uses pair-wise merge, we do it channel-by-channel via Core
        // Actually our native wrapper doesn't have a vector merge — use iterative approach
        if (mv == null || mv.isEmpty()) {
            throw new CvException("merge: input list is empty");
        }
        // Use the native merge which accepts two mats
        // Our C wrapper has opencv_core_hconcat/vconcat but not multi-merge
        // We need to handle this at the Java level or add to native
        // For now, throw if not supported
        throw new UnsupportedOperationException("merge with List not yet supported - use Mat operations directly");
    }

    public static void split(Mat src, List<Mat> mv) {
        throw new UnsupportedOperationException("split with List not yet supported - use Mat operations directly");
    }

    // ========================================================================
    // Error utility
    // ========================================================================

    private static void checkError(int result) {
        if (result < 0) {
            String msg = getLastError();
            throw new CvException(msg != null ? msg : "Core operation failed with code: " + result);
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

    /**
     * Simple wrapper for double array data, used by meanStdDev.
     */
    public static class MatOfDouble {
        private double[] data;

        public MatOfDouble() {
            data = new double[0];
        }

        public void fromArray(double[] arr) {
            data = arr.clone();
        }

        public double[] toArray() {
            return data.clone();
        }

        public double get(int idx) {
            return data[idx];
        }

        public int size() {
            return data.length;
        }
    }
}
