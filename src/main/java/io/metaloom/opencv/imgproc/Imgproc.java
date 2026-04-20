package io.metaloom.opencv.imgproc;

import io.metaloom.opencv.NativeBindings;
import io.metaloom.opencv.core.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import static io.metaloom.opencv.core.Mat.invoke;
import static io.metaloom.opencv.core.Mat.rethrow;
import static java.lang.foreign.ValueLayout.*;

public class Imgproc {

    // ========================================================================
    // Color conversion codes (most common)
    // ========================================================================
    public static final int COLOR_BGR2BGRA    = 0;
    public static final int COLOR_BGRA2BGR    = 1;
    public static final int COLOR_BGR2RGBA    = 2;
    public static final int COLOR_RGBA2BGR    = 3;
    public static final int COLOR_BGR2RGB     = 4;
    public static final int COLOR_RGB2BGR     = COLOR_BGR2RGB;
    public static final int COLOR_BGRA2RGBA   = 5;
    public static final int COLOR_RGBA2BGRA   = COLOR_BGRA2RGBA;
    public static final int COLOR_BGR2GRAY    = 6;
    public static final int COLOR_RGB2GRAY    = 7;
    public static final int COLOR_GRAY2BGR    = 8;
    public static final int COLOR_GRAY2RGB    = COLOR_GRAY2BGR;
    public static final int COLOR_GRAY2BGRA   = 9;
    public static final int COLOR_GRAY2RGBA   = COLOR_GRAY2BGRA;
    public static final int COLOR_BGRA2GRAY   = 10;
    public static final int COLOR_RGBA2GRAY   = 11;
    public static final int COLOR_BGR2BGR565  = 12;
    public static final int COLOR_RGB2BGR565  = 13;
    public static final int COLOR_BGR5652BGR  = 14;
    public static final int COLOR_BGR5652RGB  = 15;
    public static final int COLOR_BGRA2BGR565 = 16;
    public static final int COLOR_RGBA2BGR565 = 17;
    public static final int COLOR_BGR5652BGRA = 18;
    public static final int COLOR_BGR5652RGBA = 19;
    public static final int COLOR_BGR2HSV     = 40;
    public static final int COLOR_RGB2HSV     = 41;
    public static final int COLOR_BGR2Lab     = 44;
    public static final int COLOR_RGB2Lab     = 45;
    public static final int COLOR_BGR2Luv     = 50;
    public static final int COLOR_RGB2Luv     = 51;
    public static final int COLOR_BGR2HLS     = 52;
    public static final int COLOR_RGB2HLS     = 53;
    public static final int COLOR_HSV2BGR     = 54;
    public static final int COLOR_HSV2RGB     = 55;
    public static final int COLOR_Lab2BGR     = 56;
    public static final int COLOR_Lab2RGB     = 57;
    public static final int COLOR_Luv2BGR     = 58;
    public static final int COLOR_Luv2RGB     = 59;
    public static final int COLOR_HLS2BGR     = 60;
    public static final int COLOR_HLS2RGB     = 61;
    public static final int COLOR_BGR2HSV_FULL  = 66;
    public static final int COLOR_RGB2HSV_FULL  = 67;
    public static final int COLOR_BGR2HLS_FULL  = 68;
    public static final int COLOR_RGB2HLS_FULL  = 69;
    public static final int COLOR_HSV2BGR_FULL  = 70;
    public static final int COLOR_HSV2RGB_FULL  = 71;
    public static final int COLOR_HLS2BGR_FULL  = 72;
    public static final int COLOR_HLS2RGB_FULL  = 73;
    public static final int COLOR_BGR2YUV     = 82;
    public static final int COLOR_RGB2YUV     = 83;
    public static final int COLOR_YUV2BGR     = 84;
    public static final int COLOR_YUV2RGB     = 85;
    public static final int COLOR_BGR2YCrCb   = 36;
    public static final int COLOR_RGB2YCrCb   = 37;
    public static final int COLOR_YCrCb2BGR   = 38;
    public static final int COLOR_YCrCb2RGB   = 39;

    // ========================================================================
    // Threshold types
    // ========================================================================
    public static final int THRESH_BINARY     = 0;
    public static final int THRESH_BINARY_INV = 1;
    public static final int THRESH_TRUNC      = 2;
    public static final int THRESH_TOZERO     = 3;
    public static final int THRESH_TOZERO_INV = 4;
    public static final int THRESH_MASK       = 7;
    public static final int THRESH_OTSU       = 8;
    public static final int THRESH_TRIANGLE   = 16;

    // ========================================================================
    // Adaptive threshold types
    // ========================================================================
    public static final int ADAPTIVE_THRESH_MEAN_C     = 0;
    public static final int ADAPTIVE_THRESH_GAUSSIAN_C = 1;

    // ========================================================================
    // Morphological operation types
    // ========================================================================
    public static final int MORPH_ERODE    = 0;
    public static final int MORPH_DILATE   = 1;
    public static final int MORPH_OPEN     = 2;
    public static final int MORPH_CLOSE    = 3;
    public static final int MORPH_GRADIENT = 4;
    public static final int MORPH_TOPHAT   = 5;
    public static final int MORPH_BLACKHAT = 6;
    public static final int MORPH_HITMISS  = 7;

    // ========================================================================
    // Morphological shapes
    // ========================================================================
    public static final int MORPH_RECT    = 0;
    public static final int MORPH_CROSS   = 1;
    public static final int MORPH_ELLIPSE = 2;

    // ========================================================================
    // Interpolation flags
    // ========================================================================
    public static final int INTER_NEAREST        = 0;
    public static final int INTER_LINEAR         = 1;
    public static final int INTER_CUBIC          = 2;
    public static final int INTER_AREA           = 3;
    public static final int INTER_LANCZOS4       = 4;
    public static final int INTER_LINEAR_EXACT   = 5;
    public static final int INTER_NEAREST_EXACT  = 6;
    public static final int INTER_MAX            = 7;
    public static final int WARP_FILL_OUTLIERS   = 8;
    public static final int WARP_INVERSE_MAP     = 16;

    // ========================================================================
    // Line types
    // ========================================================================
    public static final int FILLED = -1;
    public static final int LINE_4 = 4;
    public static final int LINE_8 = 8;
    public static final int LINE_AA = 16;

    // ========================================================================
    // Font faces
    // ========================================================================
    public static final int FONT_HERSHEY_SIMPLEX        = 0;
    public static final int FONT_HERSHEY_PLAIN          = 1;
    public static final int FONT_HERSHEY_DUPLEX         = 2;
    public static final int FONT_HERSHEY_COMPLEX        = 3;
    public static final int FONT_HERSHEY_TRIPLEX        = 4;
    public static final int FONT_HERSHEY_COMPLEX_SMALL  = 5;
    public static final int FONT_HERSHEY_SCRIPT_SIMPLEX = 6;
    public static final int FONT_HERSHEY_SCRIPT_COMPLEX = 7;
    public static final int FONT_ITALIC                 = 16;

    // ========================================================================
    // Contour retrieval modes
    // ========================================================================
    public static final int RETR_EXTERNAL  = 0;
    public static final int RETR_LIST      = 1;
    public static final int RETR_CCOMP     = 2;
    public static final int RETR_TREE      = 3;
    public static final int RETR_FLOODFILL = 4;

    // ========================================================================
    // Contour approximation methods
    // ========================================================================
    public static final int CHAIN_APPROX_NONE      = 1;
    public static final int CHAIN_APPROX_SIMPLE    = 2;
    public static final int CHAIN_APPROX_TC89_L1   = 3;
    public static final int CHAIN_APPROX_TC89_KCOS = 4;

    private Imgproc() {
    }

    // ========================================================================
    // Color conversion
    // ========================================================================

    public static void cvtColor(Mat src, Mat dst, int code, int dstCn) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_CVT_COLOR.invokeExact(
                src.nativePtr(), dst.nativePtr(), code, dstCn));
        checkError(res);
    }

    public static void cvtColor(Mat src, Mat dst, int code) {
        cvtColor(src, dst, code, 0);
    }

    // ========================================================================
    // Resize / Geometric transforms
    // ========================================================================

    public static void resize(Mat src, Mat dst, Size dsize, double fx, double fy, int interpolation) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_RESIZE.invokeExact(
                src.nativePtr(), dst.nativePtr(), dsize.width, dsize.height, fx, fy, interpolation));
        checkError(res);
    }

    public static void resize(Mat src, Mat dst, Size dsize) {
        resize(src, dst, dsize, 0, 0, INTER_LINEAR);
    }

    public static void warpAffine(Mat src, Mat dst, Mat M, Size dsize, int flags, int borderMode, Scalar borderValue) {
        double bv0 = borderValue != null ? borderValue.val[0] : 0;
        double bv1 = borderValue != null ? borderValue.val[1] : 0;
        double bv2 = borderValue != null ? borderValue.val[2] : 0;
        double bv3 = borderValue != null ? borderValue.val[3] : 0;
        int res = invoke(() -> (int) NativeBindings.IMGPROC_WARP_AFFINE.invokeExact(
                src.nativePtr(), dst.nativePtr(), M.nativePtr(),
                dsize.width, dsize.height, flags, borderMode, bv0, bv1, bv2, bv3));
        checkError(res);
    }

    public static void warpAffine(Mat src, Mat dst, Mat M, Size dsize) {
        warpAffine(src, dst, M, dsize, INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));
    }

    public static void warpPerspective(Mat src, Mat dst, Mat M, Size dsize, int flags, int borderMode, Scalar borderValue) {
        double bv0 = borderValue != null ? borderValue.val[0] : 0;
        double bv1 = borderValue != null ? borderValue.val[1] : 0;
        double bv2 = borderValue != null ? borderValue.val[2] : 0;
        double bv3 = borderValue != null ? borderValue.val[3] : 0;
        int res = invoke(() -> (int) NativeBindings.IMGPROC_WARP_PERSPECTIVE.invokeExact(
                src.nativePtr(), dst.nativePtr(), M.nativePtr(),
                dsize.width, dsize.height, flags, borderMode, bv0, bv1, bv2, bv3));
        checkError(res);
    }

    public static void warpPerspective(Mat src, Mat dst, Mat M, Size dsize) {
        warpPerspective(src, dst, M, dsize, INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));
    }

    public static Mat getRotationMatrix2D(Point center, double angle, double scale) {
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.IMGPROC_GET_ROTATION_MATRIX_2D.invokeExact(
                center.x, center.y, angle, scale)));
    }

    // ========================================================================
    // Threshold
    // ========================================================================

    public static double threshold(Mat src, Mat dst, double thresh, double maxval, int type) {
        return invoke(() -> (double) NativeBindings.IMGPROC_THRESHOLD.invokeExact(
                src.nativePtr(), dst.nativePtr(), thresh, maxval, type));
    }

    public static void adaptiveThreshold(Mat src, Mat dst, double maxValue, int adaptiveMethod,
                                          int thresholdType, int blockSize, double C) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_ADAPTIVE_THRESHOLD.invokeExact(
                src.nativePtr(), dst.nativePtr(), maxValue, adaptiveMethod, thresholdType, blockSize, C));
        checkError(res);
    }

    // ========================================================================
    // Filtering
    // ========================================================================

    public static void blur(Mat src, Mat dst, Size ksize, Point anchor, int borderType) {
        double ax = anchor != null ? anchor.x : -1;
        double ay = anchor != null ? anchor.y : -1;
        int res = invoke(() -> (int) NativeBindings.IMGPROC_BLUR.invokeExact(
                src.nativePtr(), dst.nativePtr(), ksize.width, ksize.height, ax, ay, borderType));
        checkError(res);
    }

    public static void blur(Mat src, Mat dst, Size ksize) {
        blur(src, dst, ksize, new Point(-1, -1), Core.BORDER_DEFAULT);
    }

    public static void GaussianBlur(Mat src, Mat dst, Size ksize, double sigmaX, double sigmaY, int borderType) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_GAUSSIAN_BLUR.invokeExact(
                src.nativePtr(), dst.nativePtr(), ksize.width, ksize.height, sigmaX, sigmaY, borderType));
        checkError(res);
    }

    public static void GaussianBlur(Mat src, Mat dst, Size ksize, double sigmaX) {
        GaussianBlur(src, dst, ksize, sigmaX, 0, Core.BORDER_DEFAULT);
    }

    public static void medianBlur(Mat src, Mat dst, int ksize) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_MEDIAN_BLUR.invokeExact(
                src.nativePtr(), dst.nativePtr(), ksize));
        checkError(res);
    }

    public static void bilateralFilter(Mat src, Mat dst, int d, double sigmaColor, double sigmaSpace, int borderType) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_BILATERAL_FILTER.invokeExact(
                src.nativePtr(), dst.nativePtr(), d, sigmaColor, sigmaSpace, borderType));
        checkError(res);
    }

    public static void bilateralFilter(Mat src, Mat dst, int d, double sigmaColor, double sigmaSpace) {
        bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, Core.BORDER_DEFAULT);
    }

    public static void Sobel(Mat src, Mat dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_SOBEL.invokeExact(
                src.nativePtr(), dst.nativePtr(), ddepth, dx, dy, ksize, scale, delta, borderType));
        checkError(res);
    }

    public static void Sobel(Mat src, Mat dst, int ddepth, int dx, int dy) {
        Sobel(src, dst, ddepth, dx, dy, 3, 1, 0, Core.BORDER_DEFAULT);
    }

    public static void Scharr(Mat src, Mat dst, int ddepth, int dx, int dy, double scale, double delta, int borderType) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_SCHARR.invokeExact(
                src.nativePtr(), dst.nativePtr(), ddepth, dx, dy, scale, delta, borderType));
        checkError(res);
    }

    public static void Scharr(Mat src, Mat dst, int ddepth, int dx, int dy) {
        Scharr(src, dst, ddepth, dx, dy, 1, 0, Core.BORDER_DEFAULT);
    }

    public static void Laplacian(Mat src, Mat dst, int ddepth, int ksize, double scale, double delta, int borderType) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_LAPLACIAN.invokeExact(
                src.nativePtr(), dst.nativePtr(), ddepth, ksize, scale, delta, borderType));
        checkError(res);
    }

    public static void Laplacian(Mat src, Mat dst, int ddepth) {
        Laplacian(src, dst, ddepth, 1, 1, 0, Core.BORDER_DEFAULT);
    }

    public static void Canny(Mat image, Mat edges, double threshold1, double threshold2, int apertureSize, boolean L2gradient) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_CANNY.invokeExact(
                image.nativePtr(), edges.nativePtr(), threshold1, threshold2, apertureSize, L2gradient ? 1 : 0));
        checkError(res);
    }

    public static void Canny(Mat image, Mat edges, double threshold1, double threshold2) {
        Canny(image, edges, threshold1, threshold2, 3, false);
    }

    // ========================================================================
    // Morphology
    // ========================================================================

    public static void erode(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType, Scalar borderValue) {
        double ax = anchor != null ? anchor.x : -1;
        double ay = anchor != null ? anchor.y : -1;
        double bv0 = borderValue != null ? borderValue.val[0] : 0;
        double bv1 = borderValue != null ? borderValue.val[1] : 0;
        double bv2 = borderValue != null ? borderValue.val[2] : 0;
        double bv3 = borderValue != null ? borderValue.val[3] : 0;
        int res = invoke(() -> (int) NativeBindings.IMGPROC_ERODE.invokeExact(
                src.nativePtr(), dst.nativePtr(), kernel.nativePtr(),
                ax, ay, iterations, borderType, bv0, bv1, bv2, bv3));
        checkError(res);
    }

    public static void erode(Mat src, Mat dst, Mat kernel) {
        erode(src, dst, kernel, new Point(-1, -1), 1, Core.BORDER_CONSTANT, null);
    }

    public static void dilate(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType, Scalar borderValue) {
        double ax = anchor != null ? anchor.x : -1;
        double ay = anchor != null ? anchor.y : -1;
        double bv0 = borderValue != null ? borderValue.val[0] : 0;
        double bv1 = borderValue != null ? borderValue.val[1] : 0;
        double bv2 = borderValue != null ? borderValue.val[2] : 0;
        double bv3 = borderValue != null ? borderValue.val[3] : 0;
        int res = invoke(() -> (int) NativeBindings.IMGPROC_DILATE.invokeExact(
                src.nativePtr(), dst.nativePtr(), kernel.nativePtr(),
                ax, ay, iterations, borderType, bv0, bv1, bv2, bv3));
        checkError(res);
    }

    public static void dilate(Mat src, Mat dst, Mat kernel) {
        dilate(src, dst, kernel, new Point(-1, -1), 1, Core.BORDER_CONSTANT, null);
    }

    public static void morphologyEx(Mat src, Mat dst, int op, Mat kernel, Point anchor, int iterations,
                                     int borderType, Scalar borderValue) {
        double ax = anchor != null ? anchor.x : -1;
        double ay = anchor != null ? anchor.y : -1;
        double bv0 = borderValue != null ? borderValue.val[0] : 0;
        double bv1 = borderValue != null ? borderValue.val[1] : 0;
        double bv2 = borderValue != null ? borderValue.val[2] : 0;
        double bv3 = borderValue != null ? borderValue.val[3] : 0;
        int res = invoke(() -> (int) NativeBindings.IMGPROC_MORPHOLOGY_EX.invokeExact(
                src.nativePtr(), dst.nativePtr(), op, kernel.nativePtr(),
                ax, ay, iterations, borderType, bv0, bv1, bv2, bv3));
        checkError(res);
    }

    public static void morphologyEx(Mat src, Mat dst, int op, Mat kernel) {
        morphologyEx(src, dst, op, kernel, new Point(-1, -1), 1, Core.BORDER_CONSTANT, null);
    }

    public static Mat getStructuringElement(int shape, Size ksize, Point anchor) {
        double ax = anchor != null ? anchor.x : -1;
        double ay = anchor != null ? anchor.y : -1;
        return new Mat(invoke(() -> (MemorySegment) NativeBindings.IMGPROC_GET_STRUCTURING_ELEMENT.invokeExact(
                shape, ksize.width, ksize.height, ax, ay)));
    }

    public static Mat getStructuringElement(int shape, Size ksize) {
        return getStructuringElement(shape, ksize, new Point(-1, -1));
    }

    // ========================================================================
    // Drawing
    // ========================================================================

    public static void line(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int lineType, int shift) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_LINE.invokeExact(
                img.nativePtr(), pt1.x, pt1.y, pt2.x, pt2.y,
                color.val[0], color.val[1], color.val[2], color.val[3],
                thickness, lineType, shift));
        checkError(res);
    }

    public static void line(Mat img, Point pt1, Point pt2, Scalar color, int thickness) {
        line(img, pt1, pt2, color, thickness, LINE_8, 0);
    }

    public static void line(Mat img, Point pt1, Point pt2, Scalar color) {
        line(img, pt1, pt2, color, 1, LINE_8, 0);
    }

    public static void rectangle(Mat img, Point pt1, Point pt2, Scalar color, int thickness, int lineType, int shift) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_RECTANGLE.invokeExact(
                img.nativePtr(), pt1.x, pt1.y, pt2.x, pt2.y,
                color.val[0], color.val[1], color.val[2], color.val[3],
                thickness, lineType, shift));
        checkError(res);
    }

    public static void rectangle(Mat img, Point pt1, Point pt2, Scalar color, int thickness) {
        rectangle(img, pt1, pt2, color, thickness, LINE_8, 0);
    }

    public static void rectangle(Mat img, Point pt1, Point pt2, Scalar color) {
        rectangle(img, pt1, pt2, color, 1, LINE_8, 0);
    }

    public static void rectangle(Mat img, Rect rect, Scalar color, int thickness) {
        rectangle(img, rect.tl(), rect.br(), color, thickness, LINE_8, 0);
    }

    public static void rectangle(Mat img, Rect rect, Scalar color) {
        rectangle(img, rect, color, 1);
    }

    public static void circle(Mat img, Point center, int radius, Scalar color, int thickness, int lineType, int shift) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_CIRCLE.invokeExact(
                img.nativePtr(), center.x, center.y, radius,
                color.val[0], color.val[1], color.val[2], color.val[3],
                thickness, lineType, shift));
        checkError(res);
    }

    public static void circle(Mat img, Point center, int radius, Scalar color, int thickness) {
        circle(img, center, radius, color, thickness, LINE_8, 0);
    }

    public static void circle(Mat img, Point center, int radius, Scalar color) {
        circle(img, center, radius, color, 1, LINE_8, 0);
    }

    public static void ellipse(Mat img, Point center, Size axes, double angle, double startAngle, double endAngle,
                                Scalar color, int thickness, int lineType, int shift) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_ELLIPSE.invokeExact(
                img.nativePtr(), center.x, center.y, axes.width, axes.height,
                angle, startAngle, endAngle,
                color.val[0], color.val[1], color.val[2], color.val[3],
                thickness, lineType, shift));
        checkError(res);
    }

    public static void ellipse(Mat img, Point center, Size axes, double angle, double startAngle, double endAngle,
                                Scalar color, int thickness) {
        ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, LINE_8, 0);
    }

    public static void ellipse(Mat img, Point center, Size axes, double angle, double startAngle, double endAngle,
                                Scalar color) {
        ellipse(img, center, axes, angle, startAngle, endAngle, color, 1, LINE_8, 0);
    }

    public static void putText(Mat img, String text, Point org, int fontFace, double fontScale,
                                Scalar color, int thickness, int lineType, boolean bottomLeftOrigin) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment textAddr = arena.allocateFrom(text);
            int res = (int) NativeBindings.IMGPROC_PUT_TEXT.invokeExact(
                    img.nativePtr(), textAddr, org.x, org.y, fontFace, fontScale,
                    color.val[0], color.val[1], color.val[2], color.val[3],
                    thickness, lineType, bottomLeftOrigin ? 1 : 0);
            checkError(res);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static void putText(Mat img, String text, Point org, int fontFace, double fontScale,
                                Scalar color, int thickness) {
        putText(img, text, org, fontFace, fontScale, color, thickness, LINE_8, false);
    }

    public static void putText(Mat img, String text, Point org, int fontFace, double fontScale, Scalar color) {
        putText(img, text, org, fontFace, fontScale, color, 1, LINE_8, false);
    }

    // ========================================================================
    // Feature detection
    // ========================================================================

    public static void cornerHarris(Mat src, Mat dst, int blockSize, int ksize, double k, int borderType) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_CORNER_HARRIS.invokeExact(
                src.nativePtr(), dst.nativePtr(), blockSize, ksize, k, borderType));
        checkError(res);
    }

    public static void cornerHarris(Mat src, Mat dst, int blockSize, int ksize, double k) {
        cornerHarris(src, dst, blockSize, ksize, k, Core.BORDER_DEFAULT);
    }

    public static void HoughLines(Mat image, Mat lines, double rho, double theta, int threshold,
                                   double srn, double stn) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_HOUGH_LINES.invokeExact(
                image.nativePtr(), lines.nativePtr(), rho, theta, threshold, srn, stn));
        checkError(res);
    }

    public static void HoughLines(Mat image, Mat lines, double rho, double theta, int threshold) {
        HoughLines(image, lines, rho, theta, threshold, 0, 0);
    }

    public static void HoughLinesP(Mat image, Mat lines, double rho, double theta, int threshold,
                                    double minLineLength, double maxLineGap) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_HOUGH_LINES_P.invokeExact(
                image.nativePtr(), lines.nativePtr(), rho, theta, threshold, minLineLength, maxLineGap));
        checkError(res);
    }

    public static void HoughLinesP(Mat image, Mat lines, double rho, double theta, int threshold) {
        HoughLinesP(image, lines, rho, theta, threshold, 0, 0);
    }

    public static void getRectSubPix(Mat image, Size patchSize, Point center, Mat patch, int patchType) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_GET_RECT_SUB_PIX.invokeExact(
                image.nativePtr(), patchSize.width, patchSize.height, center.x, center.y,
                patch.nativePtr(), patchType));
        checkError(res);
    }

    public static void getRectSubPix(Mat image, Size patchSize, Point center, Mat patch) {
        getRectSubPix(image, patchSize, center, patch, -1);
    }

    // ========================================================================
    // Contours
    // ========================================================================

    public static void findContours(Mat image, List<Mat> contours, Mat hierarchy, int mode, int method, Point offset) {
        // The native wrapper returns contours in a single Mat with hierarchy
        // We need to call the native function, then parse the results
        double ox = offset != null ? offset.x : 0;
        double oy = offset != null ? offset.y : 0;
        try (Mat contoursMat = new Mat()) {
            int res = (int) NativeBindings.IMGPROC_FIND_CONTOURS.invokeExact(
                    image.nativePtr(), contoursMat.nativePtr(), hierarchy.nativePtr(),
                    mode, method, ox, oy);
            checkError(res);
            // The native function puts contours as rows of the contoursMat
            // Each row is a contour. We need to extract them.
            // For now the C wrapper stores contours as a vector of Mats - but our simple wrapper
            // just stores the raw contour data. This would need a more complex native interface.
            // TODO: Implement proper contour extraction from native wrapper
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static void findContours(Mat image, List<Mat> contours, Mat hierarchy, int mode, int method) {
        findContours(image, contours, hierarchy, mode, method, new Point(0, 0));
    }

    public static double contourArea(Mat contour, boolean oriented) {
        return invoke(() -> (double) NativeBindings.IMGPROC_CONTOUR_AREA.invokeExact(
                contour.nativePtr(), oriented ? 1 : 0));
    }

    public static double contourArea(Mat contour) {
        return contourArea(contour, false);
    }

    public static double arcLength(Mat curve, boolean closed) {
        return invoke(() -> (double) NativeBindings.IMGPROC_ARC_LENGTH.invokeExact(
                curve.nativePtr(), closed ? 1 : 0));
    }

    public static Rect boundingRect(Mat array) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment xp = arena.allocate(JAVA_INT);
            MemorySegment yp = arena.allocate(JAVA_INT);
            MemorySegment wp = arena.allocate(JAVA_INT);
            MemorySegment hp = arena.allocate(JAVA_INT);
            int res = (int) NativeBindings.IMGPROC_BOUNDING_RECT.invokeExact(
                    array.nativePtr(), xp, yp, wp, hp);
            checkError(res);
            return new Rect(xp.get(JAVA_INT, 0), yp.get(JAVA_INT, 0),
                    wp.get(JAVA_INT, 0), hp.get(JAVA_INT, 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ========================================================================
    // Histogram
    // ========================================================================

    public static void equalizeHist(Mat src, Mat dst) {
        int res = invoke(() -> (int) NativeBindings.IMGPROC_EQUALIZE_HIST.invokeExact(
                src.nativePtr(), dst.nativePtr()));
        checkError(res);
    }

    // ========================================================================
    // Error utility
    // ========================================================================

    private static void checkError(int result) {
        if (result < 0) {
            String msg = getLastError();
            throw new CvException(msg != null ? msg : "Imgproc operation failed with code: " + result);
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
