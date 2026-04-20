package io.metaloom.opencv.core;

import io.metaloom.opencv.NativeBindings;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

import static java.lang.foreign.ValueLayout.*;

/**
 * FFM-backed wrapper around a native cv::Mat.
 * Implements AutoCloseable for deterministic native memory management.
 */
public class Mat implements AutoCloseable {

    private MemorySegment nativePtr;

    // ========================================================================
    // Constructors
    // ========================================================================

    /**
     * Wrap an existing native Mat pointer. The Mat takes ownership.
     */
    public Mat(MemorySegment nativePtr) {
        if (nativePtr == null || nativePtr.equals(MemorySegment.NULL)) {
            throw new CvException("Failed to create Mat: null native pointer");
        }
        this.nativePtr = nativePtr;
    }

    /** Create an empty Mat. */
    public Mat() {
        this(invoke(() -> (MemorySegment) NativeBindings.MAT_CREATE.invokeExact()));
    }

    /** Create a Mat with given rows, cols, and type. */
    public Mat(int rows, int cols, int type) {
        this(invoke(() -> (MemorySegment) NativeBindings.MAT_CREATE_WITH_SIZE.invokeExact(rows, cols, type)));
    }

    /** Create a Mat with given rows, cols, type and initial scalar value. */
    public Mat(int rows, int cols, int type, Scalar s) {
        this(invoke(() -> (MemorySegment) NativeBindings.MAT_CREATE_WITH_SCALAR.invokeExact(
                rows, cols, type, s.val[0], s.val[1], s.val[2], s.val[3])));
    }

    // ========================================================================
    // Resource management
    // ========================================================================

    public MemorySegment nativePtr() {
        checkReleased();
        return nativePtr;
    }

    @Override
    public void close() {
        if (nativePtr != null && !nativePtr.equals(MemorySegment.NULL)) {
            try {
                NativeBindings.MAT_DELETE.invokeExact(nativePtr);
            } catch (Throwable t) {
                throw rethrow(t);
            }
            nativePtr = MemorySegment.NULL;
        }
    }

    public void release() {
        if (nativePtr != null && !nativePtr.equals(MemorySegment.NULL)) {
            try {
                NativeBindings.MAT_RELEASE.invokeExact(nativePtr);
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }
    }

    private void checkReleased() {
        if (nativePtr == null || nativePtr.equals(MemorySegment.NULL)) {
            throw new CvException("Mat has been released");
        }
    }

    // ========================================================================
    // Metadata
    // ========================================================================

    public int rows() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_ROWS.invokeExact(nativePtr));
    }

    public int cols() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_COLS.invokeExact(nativePtr));
    }

    public int type() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_TYPE.invokeExact(nativePtr));
    }

    public int depth() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_DEPTH.invokeExact(nativePtr));
    }

    public int channels() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_CHANNELS.invokeExact(nativePtr));
    }

    public int dims() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_DIMS.invokeExact(nativePtr));
    }

    public boolean empty() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_EMPTY.invokeExact(nativePtr)) != 0;
    }

    public long total() {
        checkReleased();
        return invoke(() -> (long) NativeBindings.MAT_TOTAL.invokeExact(nativePtr));
    }

    public long elemSize() {
        checkReleased();
        return invoke(() -> (long) NativeBindings.MAT_ELEM_SIZE.invokeExact(nativePtr));
    }

    public long elemSize1() {
        checkReleased();
        return invoke(() -> (long) NativeBindings.MAT_ELEM_SIZE1.invokeExact(nativePtr));
    }

    public long step1(int i) {
        checkReleased();
        return invoke(() -> (long) NativeBindings.MAT_STEP1.invokeExact(nativePtr, i));
    }

    public long step1() {
        return step1(0);
    }

    public MemorySegment dataAddr() {
        checkReleased();
        return invoke(() -> (MemorySegment) NativeBindings.MAT_DATA_ADDR.invokeExact(nativePtr));
    }

    public boolean isContinuous() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_IS_CONTINUOUS.invokeExact(nativePtr)) != 0;
    }

    public boolean isSubmatrix() {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_IS_SUBMATRIX.invokeExact(nativePtr)) != 0;
    }

    public Size size() {
        return new Size(cols(), rows());
    }

    public int width() {
        return cols();
    }

    public int height() {
        return rows();
    }

    // ========================================================================
    // Data access – put
    // ========================================================================

    public int put(int row, int col, byte[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocateFrom(JAVA_BYTE, data);
            int res = (int) NativeBindings.MAT_PUT_B.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int put(int row, int col, short[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocateFrom(JAVA_SHORT, data);
            int res = (int) NativeBindings.MAT_PUT_S.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int put(int row, int col, int[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocateFrom(JAVA_INT, data);
            int res = (int) NativeBindings.MAT_PUT_I.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int put(int row, int col, float[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocateFrom(JAVA_FLOAT, data);
            int res = (int) NativeBindings.MAT_PUT_F.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int put(int row, int col, double[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocateFrom(JAVA_DOUBLE, data);
            int res = (int) NativeBindings.MAT_PUT_D.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ========================================================================
    // Data access – get
    // ========================================================================

    public int get(int row, int col, byte[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(JAVA_BYTE, data.length);
            int res = (int) NativeBindings.MAT_GET_B.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            MemorySegment.copy(buf, JAVA_BYTE, 0, data, 0, data.length);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int get(int row, int col, short[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(JAVA_SHORT, data.length);
            int res = (int) NativeBindings.MAT_GET_S.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            MemorySegment.copy(buf, JAVA_SHORT, 0, data, 0, data.length);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int get(int row, int col, int[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(JAVA_INT, data.length);
            int res = (int) NativeBindings.MAT_GET_I.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            MemorySegment.copy(buf, JAVA_INT, 0, data, 0, data.length);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int get(int row, int col, float[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(JAVA_FLOAT, data.length);
            int res = (int) NativeBindings.MAT_GET_F.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            MemorySegment.copy(buf, JAVA_FLOAT, 0, data, 0, data.length);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int get(int row, int col, double[] data) {
        checkReleased();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(JAVA_DOUBLE, data.length);
            int res = (int) NativeBindings.MAT_GET_D.invokeExact(nativePtr, row, col, data.length, buf);
            checkNativeError(res);
            MemorySegment.copy(buf, JAVA_DOUBLE, 0, data, 0, data.length);
            return res;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Get a single pixel's values as doubles. Returns one double per channel.
     */
    public double[] get(int row, int col) {
        checkReleased();
        int ch = channels();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(JAVA_DOUBLE, ch);
            int res = (int) NativeBindings.MAT_GET_PIXEL.invokeExact(nativePtr, row, col, buf);
            if (res < 0) {
                return null;
            }
            double[] result = new double[ch];
            MemorySegment.copy(buf, JAVA_DOUBLE, 0, result, 0, ch);
            return result;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ========================================================================
    // Mat operations
    // ========================================================================

    public Mat clone() {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_CLONE.invokeExact(nativePtr)));
    }

    public void copyTo(Mat dst) {
        checkReleased();
        int res = invoke(() -> (int) NativeBindings.MAT_COPY_TO.invokeExact(nativePtr, dst.nativePtr()));
        checkNativeError(res);
    }

    public void copyTo(Mat dst, Mat mask) {
        checkReleased();
        int res = invoke(() -> (int) NativeBindings.MAT_COPY_TO_MASKED.invokeExact(
                nativePtr, dst.nativePtr(), mask.nativePtr()));
        checkNativeError(res);
    }

    public void convertTo(Mat dst, int rtype) {
        convertTo(dst, rtype, 1.0, 0.0);
    }

    public void convertTo(Mat dst, int rtype, double alpha, double beta) {
        checkReleased();
        int res = invoke(() -> (int) NativeBindings.MAT_CONVERT_TO.invokeExact(
                nativePtr, dst.nativePtr(), rtype, alpha, beta));
        checkNativeError(res);
    }

    public Mat setTo(Scalar s) {
        checkReleased();
        invoke(() -> (MemorySegment) NativeBindings.MAT_SET_TO_SCALAR.invokeExact(
                nativePtr, s.val[0], s.val[1], s.val[2], s.val[3]));
        return this;
    }

    public Mat setTo(Scalar s, Mat mask) {
        checkReleased();
        invoke(() -> (MemorySegment) NativeBindings.MAT_SET_TO_MASKED.invokeExact(
                nativePtr, s.val[0], s.val[1], s.val[2], s.val[3], mask.nativePtr()));
        return this;
    }

    public Mat reshape(int cn, int rows) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_RESHAPE.invokeExact(nativePtr, cn, rows)));
    }

    public Mat reshape(int cn) {
        return reshape(cn, 0);
    }

    public Mat row(int y) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_ROW.invokeExact(nativePtr, y)));
    }

    public Mat col(int x) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_COL.invokeExact(nativePtr, x)));
    }

    public Mat rowRange(int startrow, int endrow) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_ROW_RANGE.invokeExact(nativePtr, startrow, endrow)));
    }

    public Mat rowRange(Range r) {
        return rowRange(r.start, r.end);
    }

    public Mat colRange(int startcol, int endcol) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_COL_RANGE.invokeExact(nativePtr, startcol, endcol)));
    }

    public Mat colRange(Range r) {
        return colRange(r.start, r.end);
    }

    public Mat submat(int rowStart, int rowEnd, int colStart, int colEnd) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_SUBMAT.invokeExact(
                nativePtr, rowStart, rowEnd, colStart, colEnd)));
    }

    public Mat submat(Range rowRange, Range colRange) {
        return submat(rowRange.start, rowRange.end, colRange.start, colRange.end);
    }

    public Mat submat(Rect roi) {
        return submat(roi.y, roi.y + roi.height, roi.x, roi.x + roi.width);
    }

    public Mat t() {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_T.invokeExact(nativePtr)));
    }

    public Mat inv(int method) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_INV.invokeExact(nativePtr, method)));
    }

    public Mat inv() {
        return inv(0); // DECOMP_LU
    }

    public Mat mul(Mat m, double scale) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_MUL.invokeExact(nativePtr, m.nativePtr(), scale)));
    }

    public Mat mul(Mat m) {
        return mul(m, 1.0);
    }

    public Mat matMul(Mat m) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_MAT_MUL.invokeExact(nativePtr, m.nativePtr())));
    }

    public double dot(Mat m) {
        checkReleased();
        return invoke(() -> (double) NativeBindings.MAT_DOT.invokeExact(nativePtr, m.nativePtr()));
    }

    public Mat cross(Mat m) {
        checkReleased();
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_CROSS.invokeExact(nativePtr, m.nativePtr())));
    }

    public void push_back(Mat m) {
        checkReleased();
        int res = invoke(() -> (int) NativeBindings.MAT_PUSH_BACK.invokeExact(nativePtr, m.nativePtr()));
        checkNativeError(res);
    }

    public void create(int rows, int cols, int type) {
        checkReleased();
        try {
            NativeBindings.MAT_CREATE_INPLACE.invokeExact(nativePtr, rows, cols, type);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int checkVector(int elemChannels, int depth, boolean requireContinuous) {
        checkReleased();
        return invoke(() -> (int) NativeBindings.MAT_CHECK_VECTOR.invokeExact(
                nativePtr, elemChannels, depth, requireContinuous ? 1 : 0));
    }

    public int checkVector(int elemChannels) {
        return checkVector(elemChannels, -1, true);
    }

    // ========================================================================
    // Static factories
    // ========================================================================

    public static Mat zeros(int rows, int cols, int type) {
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_ZEROS.invokeExact(rows, cols, type)));
    }

    public static Mat zeros(Size size, int type) {
        return zeros((int) size.height, (int) size.width, type);
    }

    public static Mat ones(int rows, int cols, int type) {
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_ONES.invokeExact(rows, cols, type)));
    }

    public static Mat eye(int rows, int cols, int type) {
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_EYE.invokeExact(rows, cols, type)));
    }

    public static Mat diag(Mat d) {
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.MAT_DIAG.invokeExact(d.nativePtr(), 0)));
    }

    // ========================================================================
    // Utility
    // ========================================================================

    public String dump() {
        checkReleased();
        MemorySegment str = invoke(() -> (MemorySegment) NativeBindings.MAT_DUMP.invokeExact(nativePtr));
        if (str.equals(MemorySegment.NULL)) {
            return "";
        }
        return str.reinterpret(4096).getString(0);
    }

    @Override
    public String toString() {
        if (nativePtr == null || nativePtr.equals(MemorySegment.NULL)) {
            return "Mat [ released ]";
        }
        return "Mat [ " + rows() + "*" + cols() + "*" + CvType.typeToString(type())
                + ", isCont=" + isContinuous()
                + ", isSubmat=" + isSubmatrix()
                + ", nativePtr=0x" + Long.toHexString(nativePtr.address())
                + " ]";
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    private static void checkNativeError(int result) {
        if (result < 0) {
            String errorMsg = getLastError();
            throw new CvException(errorMsg != null ? errorMsg : "Native operation failed with code: " + result);
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

    @FunctionalInterface
    public interface NativeCall<T> {
        T call() throws Throwable;
    }

    public static <T> T invoke(NativeCall<T> call) {
        try {
            return call.call();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException re) throw re;
        if (t instanceof Error e) throw e;
        throw new CvException(t.getMessage());
    }
}
