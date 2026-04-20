package io.metaloom.opencv.videoio;

import io.metaloom.opencv.core.*;
import io.metaloom.opencv.imgproc.Imgproc;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests exercising the OpenCV FFM APIs used by video4j:
 * - VideoCapture lifecycle
 * - Mat get/put for BufferedImage conversion
 * - Imgproc.resize with different interpolation modes
 * - Imgproc.cvtColor (grayscale conversions)
 * - Imgproc.blur
 * - Core.copyMakeBorder (boxing/letterboxing)
 * - Core.normalize / Core.convertScaleAbs
 * - Core.mean (black frame detection)
 * - Imgproc.Laplacian (blurriness detection)
 * - Core.sqrt
 * - Imgproc.cornerHarris
 * - Imgproc.Canny + Imgproc.HoughLinesP
 * - Imgproc.getRectSubPix
 * - Imgproc.putText / rectangle / circle / line drawing
 * - Mat.submat (crop operations)
 * - Mat.convertTo (contrast adjustment)
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class Video4jApiTest {

    // ========================================================================
    // VideoCapture
    // ========================================================================

    @Test
    @Order(1)
    void testVideoCaptureCreate() {
        try (VideoCapture cap = new VideoCapture()) {
            assertNotNull(cap);
            assertFalse(cap.isOpened());
        }
    }

    @Test
    @Order(2)
    void testVideoCaptureOpenNonExistent() {
        try (VideoCapture cap = new VideoCapture()) {
            boolean opened = cap.open("/nonexistent/video.mp4");
            assertFalse(opened);
            assertFalse(cap.isOpened());
        }
    }

    @Test
    @Order(3)
    void testVideoCaptureRelease() {
        try (VideoCapture cap = new VideoCapture()) {
            cap.release();
            assertFalse(cap.isOpened());
        }
    }

    @Test
    @Order(4)
    void testVideoCaptureSetGet() {
        // Even without an opened source, set/get should not crash
        try (VideoCapture cap = new VideoCapture()) {
            // These will return 0 since nothing is opened, but should not throw
            double val = cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
            assertEquals(0.0, val);
        }
    }

    // ========================================================================
    // Mat get/put for BufferedImage conversion (like CVUtils.matToBufferedImage)
    // ========================================================================

    @Test
    @Order(10)
    void testMatGetPutByteArray() {
        // Simulates creating a frame and reading its pixel data (like matToBufferedImage)
        int width = 320;
        int height = 240;
        int channels = 3;
        try (Mat mat = new Mat(height, width, CvType.CV_8UC3)) {
            // Write pixel data
            byte[] pixels = new byte[width * height * channels];
            for (int i = 0; i < pixels.length; i++) {
                pixels[i] = (byte) (i % 256);
            }
            int written = mat.put(0, 0, pixels);
            assertEquals(pixels.length, written);

            // Read back
            byte[] readBack = new byte[pixels.length];
            int read = mat.get(0, 0, readBack);
            assertEquals(pixels.length, read);
            assertArrayEquals(pixels, readBack);
        }
    }

    @Test
    @Order(11)
    void testMatSinglePixelGet() {
        try (Mat mat = new Mat(2, 2, CvType.CV_8UC3, new Scalar(100, 150, 200))) {
            double[] pixel = mat.get(0, 0);
            assertNotNull(pixel);
            assertEquals(3, pixel.length);
            assertEquals(100, pixel[0], 1e-5);
            assertEquals(150, pixel[1], 1e-5);
            assertEquals(200, pixel[2], 1e-5);
        }
    }

    // ========================================================================
    // Imgproc.resize (used by CVUtils.resize and ExtendedVideoCapture.read)
    // ========================================================================

    @Test
    @Order(20)
    void testResizeWithLanczos() {
        try (Mat src = new Mat(480, 640, CvType.CV_8UC3, new Scalar(128, 128, 128));
             Mat dst = new Mat()) {
            Imgproc.resize(src, dst, new Size(320, 240), 0, 0, Imgproc.INTER_LANCZOS4);
            assertEquals(240, dst.rows());
            assertEquals(320, dst.cols());
            assertEquals(CvType.CV_8UC3, dst.type());
        }
    }

    @Test
    @Order(21)
    void testResizeInPlace() {
        // video4j sometimes resizes in-place: Imgproc.resize(target, target, ...)
        try (Mat mat = new Mat(100, 200, CvType.CV_8UC3, new Scalar(64, 64, 64))) {
            Imgproc.resize(mat, mat, new Size(50, 25), 0, 0, Imgproc.INTER_LINEAR);
            assertEquals(25, mat.rows());
            assertEquals(50, mat.cols());
        }
    }

    // ========================================================================
    // Core.copyMakeBorder (used by boxFrame / letterboxing)
    // ========================================================================

    @Test
    @Order(30)
    void testCopyMakeBorder() {
        int resX = 256;
        int resY = 200;
        int spaceY = (resX - resY) / 2; // 28
        try (Mat src = new Mat(resY, resX, CvType.CV_8UC3, new Scalar(128, 128, 128));
             Mat dst = new Mat()) {
            Core.copyMakeBorder(src, dst, spaceY, spaceY, 0, 0, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
            assertEquals(resX, dst.rows()); // resY + 2*spaceY = 256
            assertEquals(resX, dst.cols());

            // Verify border is black
            double[] topPixel = dst.get(0, 0);
            assertEquals(0, topPixel[0], 1e-5);
            assertEquals(0, topPixel[1], 1e-5);
            assertEquals(0, topPixel[2], 1e-5);

            // Verify content area is preserved
            double[] contentPixel = dst.get(spaceY, 0);
            assertEquals(128, contentPixel[0], 1e-5);
        }
    }

    // ========================================================================
    // cvtColor (grayscale conversions used throughout video4j)
    // ========================================================================

    @Test
    @Order(40)
    void testCvtColorRgb2Gray() {
        try (Mat rgb = new Mat(100, 100, CvType.CV_8UC3, new Scalar(100, 150, 200));
             Mat gray = new Mat()) {
            Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY);
            assertEquals(100, gray.rows());
            assertEquals(100, gray.cols());
            assertEquals(CvType.CV_8UC1, gray.type());
            assertEquals(1, gray.channels());
        }
    }

    @Test
    @Order(41)
    void testCvtColorGray2Bgr() {
        try (Mat gray = new Mat(50, 50, CvType.CV_8UC1, new Scalar(128));
             Mat bgr = new Mat()) {
            Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);
            assertEquals(CvType.CV_8UC3, bgr.type());
            double[] pixel = bgr.get(0, 0);
            assertEquals(3, pixel.length);
            assertEquals(128, pixel[0], 1e-5);
            assertEquals(128, pixel[1], 1e-5);
            assertEquals(128, pixel[2], 1e-5);
        }
    }

    // ========================================================================
    // Imgproc.blur (used by CVUtils.blur)
    // ========================================================================

    @Test
    @Order(50)
    void testBlur() {
        try (Mat src = new Mat(100, 100, CvType.CV_8UC3, new Scalar(128, 128, 128));
             Mat dst = new Mat()) {
            Imgproc.blur(src, dst, new Size(20, 20), new Point(-1, -1), Core.BORDER_DEFAULT);
            assertEquals(src.rows(), dst.rows());
            assertEquals(src.cols(), dst.cols());
            assertFalse(dst.empty());
        }
    }

    // ========================================================================
    // Core.normalize + Core.convertScaleAbs (used by CVUtils.harris)
    // ========================================================================

    @Test
    @Order(60)
    void testNormalizeAndConvertScaleAbs() {
        try (Mat src = new Mat(10, 10, CvType.CV_32FC1);
             Mat norm = new Mat();
             Mat abs = new Mat()) {
            // Fill with values 0..99
            float[] data = new float[100];
            for (int i = 0; i < 100; i++) data[i] = i;
            src.put(0, 0, data);

            Core.normalize(src, norm, 0, 255, Core.NORM_MINMAX, -1, null);
            assertFalse(norm.empty());

            Core.convertScaleAbs(norm, abs);
            assertEquals(CvType.CV_8UC1, abs.type());
        }
    }

    // ========================================================================
    // Core.mean (used by CVUtils.isBlackFrame / CVUtils.mean)
    // ========================================================================

    @Test
    @Order(70)
    void testMeanForBlackFrameDetection() {
        // Black frame
        try (Mat black = new Mat(100, 100, CvType.CV_8UC3, new Scalar(0, 0, 0))) {
            Scalar mean = Core.mean(black);
            assertEquals(0, mean.val[0], 1e-5);
            assertTrue(mean.val[0] < 10.0); // "black" threshold
        }

        // Non-black frame
        try (Mat bright = new Mat(100, 100, CvType.CV_8UC3, new Scalar(128, 128, 128))) {
            Scalar mean = Core.mean(bright);
            assertTrue(mean.val[0] > 10.0);
        }
    }

    // ========================================================================
    // Imgproc.Laplacian (used by CVUtils.blurriness)
    // ========================================================================

    @Test
    @Order(80)
    void testBlurrinessDetectionViaLaplacian() {
        try (Mat sharp = new Mat(100, 100, CvType.CV_8UC1);
             Mat blurry = new Mat();
             Mat laplacianSharp = new Mat();
             Mat laplacianBlurry = new Mat()) {
            // Create a "sharp" image with edges
            byte[] pixels = new byte[100 * 100];
            for (int r = 0; r < 100; r++) {
                for (int c = 0; c < 100; c++) {
                    pixels[r * 100 + c] = (c < 50) ? (byte) 0 : (byte) 255;
                }
            }
            sharp.put(0, 0, pixels);

            // Blur it
            Imgproc.GaussianBlur(sharp, blurry, new Size(15, 15), 5);

            // Laplacian on both
            Imgproc.Laplacian(sharp, laplacianSharp, CvType.CV_64F);
            Imgproc.Laplacian(blurry, laplacianBlurry, CvType.CV_64F);

            // Variance of Laplacian = blurriness measure
            // Use Core.meanStdDev
            Core.MatOfDouble meanSharp = new Core.MatOfDouble();
            Core.MatOfDouble stdSharp = new Core.MatOfDouble();
            Core.MatOfDouble meanBlurry = new Core.MatOfDouble();
            Core.MatOfDouble stdBlurry = new Core.MatOfDouble();

            Core.meanStdDev(laplacianSharp, meanSharp, stdSharp);
            Core.meanStdDev(laplacianBlurry, meanBlurry, stdBlurry);

            double varianceSharp = stdSharp.get(0) * stdSharp.get(0);
            double varianceBlurry = stdBlurry.get(0) * stdBlurry.get(0);

            // Sharp image should have higher Laplacian variance
            assertTrue(varianceSharp > varianceBlurry,
                    "Sharp image Laplacian variance (" + varianceSharp + ") should be > blurry image (" + varianceBlurry + ")");
        }
    }

    // ========================================================================
    // Core.sqrt (used by CVUtils.harris pipeline)
    // ========================================================================

    @Test
    @Order(85)
    void testSqrt() {
        try (Mat src = new Mat(2, 2, CvType.CV_32FC1);
             Mat dst = new Mat()) {
            src.put(0, 0, new float[]{4.0f, 9.0f, 16.0f, 25.0f});
            Core.sqrt(src, dst);
            float[] result = new float[4];
            dst.get(0, 0, result);
            assertEquals(2.0f, result[0], 0.001f);
            assertEquals(3.0f, result[1], 0.001f);
            assertEquals(4.0f, result[2], 0.001f);
            assertEquals(5.0f, result[3], 0.001f);
        }
    }

    // ========================================================================
    // Imgproc.cornerHarris (used by CVUtils.harris)
    // ========================================================================

    @Test
    @Order(90)
    void testCornerHarris() {
        try (Mat gray = new Mat(100, 100, CvType.CV_8UC1, new Scalar(128));
             Mat floatGray = new Mat();
             Mat corners = new Mat()) {
            gray.convertTo(floatGray, CvType.CV_32FC1);
            Imgproc.cornerHarris(floatGray, corners, 2, 3, 0.04);
            assertEquals(100, corners.rows());
            assertEquals(100, corners.cols());
            assertEquals(CvType.CV_32FC1, corners.type());
        }
    }

    // ========================================================================
    // Imgproc.Canny + HoughLinesP (used by CVUtils.houghLinesP)
    // ========================================================================

    @Test
    @Order(100)
    void testCannyAndHoughLinesP() {
        try (Mat img = new Mat(200, 200, CvType.CV_8UC1, new Scalar(0));
             Mat edges = new Mat();
             Mat lines = new Mat()) {
            // Draw a white line by setting pixels
            byte[] white = new byte[]{(byte) 255};
            for (int c = 10; c < 190; c++) {
                img.put(100, c, white);
            }

            Imgproc.Canny(img, edges, 50, 150);
            assertFalse(edges.empty());
            assertEquals(CvType.CV_8UC1, edges.type());

            Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, 10, 50, 10);
            // Should detect at least one line
            assertTrue(lines.rows() > 0, "HoughLinesP should detect at least one line");
        }
    }

    // ========================================================================
    // Imgproc.HoughLines
    // ========================================================================

    @Test
    @Order(101)
    void testHoughLines() {
        try (Mat img = new Mat(200, 200, CvType.CV_8UC1, new Scalar(0));
             Mat edges = new Mat();
             Mat lines = new Mat()) {
            byte[] white = new byte[]{(byte) 255};
            for (int c = 10; c < 190; c++) {
                img.put(100, c, white);
            }
            Imgproc.Canny(img, edges, 50, 150);
            Imgproc.HoughLines(edges, lines, 1, Math.PI / 180, 50);
            assertTrue(lines.rows() > 0, "HoughLines should detect at least one line");
        }
    }

    // ========================================================================
    // Imgproc.getRectSubPix (used by CVUtils.crop)
    // ========================================================================

    @Test
    @Order(110)
    void testGetRectSubPix() {
        try (Mat src = new Mat(100, 100, CvType.CV_8UC3, new Scalar(100, 150, 200));
             Mat patch = new Mat()) {
            Imgproc.getRectSubPix(src, new Size(20, 20), new Point(50, 50), patch);
            assertEquals(20, patch.rows());
            assertEquals(20, patch.cols());
            double[] pixel = patch.get(10, 10);
            assertEquals(100, pixel[0], 1e-5);
            assertEquals(150, pixel[1], 1e-5);
            assertEquals(200, pixel[2], 1e-5);
        }
    }

    // ========================================================================
    // Drawing primitives (used by CVUtils.drawText, drawRect, drawCircle)
    // ========================================================================

    @Test
    @Order(120)
    void testDrawingPrimitives() {
        try (Mat img = new Mat(200, 200, CvType.CV_8UC3, new Scalar(0, 0, 0))) {
            // putText
            Imgproc.putText(img, "Hello", new Point(10, 100),
                    Imgproc.FONT_HERSHEY_DUPLEX, 1.0, new Scalar(255, 255, 255), 2);

            // rectangle
            Imgproc.rectangle(img, new Point(10, 10), new Point(50, 50),
                    new Scalar(0, 255, 0), 2);

            // circle
            Imgproc.circle(img, new Point(100, 100), 30,
                    new Scalar(255, 0, 0), 2);

            // line
            Imgproc.line(img, new Point(0, 0), new Point(199, 199),
                    new Scalar(0, 0, 255), 1);

            // Verify image is no longer all black
            Scalar mean = Core.mean(img);
            assertTrue(mean.val[0] > 0 || mean.val[1] > 0 || mean.val[2] > 0,
                    "Image should not be all-black after drawing");
        }
    }

    @Test
    @Order(121)
    void testRectangleFromRect() {
        try (Mat img = new Mat(100, 100, CvType.CV_8UC3, new Scalar(0, 0, 0))) {
            Rect r = new Rect(10, 10, 30, 30);
            Imgproc.rectangle(img, r, new Scalar(255, 0, 0), 2);
            Scalar mean = Core.mean(img);
            assertTrue(mean.val[0] > 0, "Rectangle should have been drawn");
        }
    }

    // ========================================================================
    // Mat.submat / crop (used by CVUtils.crop)
    // ========================================================================

    @Test
    @Order(130)
    void testSubmatCrop() {
        try (Mat src = new Mat(100, 100, CvType.CV_8UC3, new Scalar(50, 100, 150))) {
            // Set a specific region to a different color
            try (Mat roi = src.submat(new Rect(20, 30, 40, 50))) {
                assertEquals(50, roi.rows());
                assertEquals(40, roi.cols());
                double[] pixel = roi.get(0, 0);
                assertEquals(50, pixel[0], 1e-5);
                assertEquals(100, pixel[1], 1e-5);
                assertEquals(150, pixel[2], 1e-5);
            }
        }
    }

    // ========================================================================
    // Mat.convertTo (used by CVUtils.increaseContrast: src.convertTo(dst, -1, alpha, beta))
    // ========================================================================

    @Test
    @Order(140)
    void testConvertToForContrast() {
        try (Mat src = new Mat(10, 10, CvType.CV_8UC1, new Scalar(100));
             Mat dst = new Mat()) {
            // alpha=1.5, beta=10 => 100*1.5+10=160
            src.convertTo(dst, -1, 1.5, 10);
            double[] val = dst.get(0, 0);
            assertEquals(160, val[0], 1.0);
        }
    }

    // ========================================================================
    // Mat.width() / Mat.height() / Mat.channels() (used everywhere in video4j)
    // ========================================================================

    @Test
    @Order(150)
    void testMatMetadataAccessors() {
        try (Mat mat = new Mat(480, 640, CvType.CV_8UC3)) {
            assertEquals(640, mat.width());
            assertEquals(480, mat.height());
            assertEquals(3, mat.channels());
            assertEquals(CvType.CV_8UC3, mat.type());
            assertFalse(mat.empty());
            assertEquals(640 * 480, mat.total());
        }
    }

    // ========================================================================
    // Mat empty constructor (used by MatProvider.mat() pattern in video4j)
    // ========================================================================

    @Test
    @Order(160)
    void testMatEmptyConstructor() {
        try (Mat mat = new Mat()) {
            assertTrue(mat.empty());
            assertEquals(0, mat.rows());
            assertEquals(0, mat.cols());
        }
    }

    // ========================================================================
    // Mat.setTo(Scalar) (used by CVUtils.clear)
    // ========================================================================

    @Test
    @Order(170)
    void testMatSetTo() {
        try (Mat mat = new Mat(10, 10, CvType.CV_8UC3, new Scalar(100, 100, 100))) {
            mat.setTo(new Scalar(42, 0, 0));
            double[] pixel = mat.get(5, 5);
            assertEquals(42, pixel[0], 1e-5);
            assertEquals(0, pixel[1], 1e-5);
            assertEquals(0, pixel[2], 1e-5);
        }
    }

    // ========================================================================
    // Scalar.all (used by video4j)
    // ========================================================================

    @Test
    @Order(175)
    void testScalarAll() {
        Scalar s = Scalar.all(42);
        assertEquals(42, s.val[0]);
        assertEquals(42, s.val[1]);
        assertEquals(42, s.val[2]);
        assertEquals(42, s.val[3]);
    }

    // ========================================================================
    // equalizeHist (used in face detection path)
    // ========================================================================

    @Test
    @Order(180)
    void testEqualizeHist() {
        try (Mat gray = new Mat(100, 100, CvType.CV_8UC1, new Scalar(100));
             Mat dst = new Mat()) {
            Imgproc.equalizeHist(gray, dst);
            assertEquals(gray.rows(), dst.rows());
            assertEquals(gray.cols(), dst.cols());
            assertEquals(CvType.CV_8UC1, dst.type());
        }
    }

    // ========================================================================
    // Full video4j pipeline simulation: resize + copyMakeBorder (boxFrame)
    // ========================================================================

    @Test
    @Order(190)
    void testBoxFramePipeline() {
        int targetResX = 256;
        int sourceWidth = 640;
        int sourceHeight = 480;
        double ratio = (double) sourceWidth / sourceHeight;
        int resY = (int) (targetResX / ratio);
        int spaceY = (targetResX - resY) / 2;

        try (Mat source = new Mat(sourceHeight, sourceWidth, CvType.CV_8UC3, new Scalar(128, 128, 128));
             Mat resized = new Mat();
             Mat boxed = new Mat()) {
            // Step 1: resize
            Imgproc.resize(source, resized, new Size(targetResX, resY), 0, 0, Imgproc.INTER_LANCZOS4);
            assertEquals(targetResX, resized.cols());
            assertEquals(resY, resized.rows());

            // Step 2: add border (letterbox)
            Core.copyMakeBorder(resized, boxed, spaceY, spaceY, 0, 0,
                    Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
            assertEquals(targetResX, boxed.cols());
            assertEquals(resY + 2 * spaceY, boxed.rows());
        }
    }

    // ========================================================================
    // Full video4j pipeline: grayscale -> laplacian -> meanStdDev (blurriness)
    // ========================================================================

    @Test
    @Order(200)
    void testBlurrinessPipeline() {
        try (Mat bgr = new Mat(100, 100, CvType.CV_8UC3, new Scalar(100, 150, 200));
             Mat gray = new Mat();
             Mat laplacian = new Mat()) {
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);

            Core.MatOfDouble mean = new Core.MatOfDouble();
            Core.MatOfDouble stddev = new Core.MatOfDouble();
            Core.meanStdDev(laplacian, mean, stddev);

            double variance = stddev.get(0) * stddev.get(0);
            assertFalse(Double.isNaN(variance));
            assertTrue(variance >= 0);
        }
    }

    // ========================================================================
    // Full video4j pipeline: harris corner detection
    // ========================================================================

    @Test
    @Order(210)
    void testHarrisCornerPipeline() {
        try (Mat rgb = new Mat(100, 100, CvType.CV_8UC3, new Scalar(128, 128, 128));
             Mat gray = new Mat();
             Mat corners = new Mat();
             Mat tempDst = new Mat();
             Mat tempDstNorm = new Mat()) {
            // Step 1: convert to grayscale
            Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY);

            // Step 2: Harris corner detection
            Imgproc.cornerHarris(gray, tempDst, 2, 3, 0.04);

            // Step 3: Normalize
            Core.normalize(tempDst, tempDstNorm, 0, 255, Core.NORM_MINMAX);

            // Step 4: Convert to absolute values
            Core.convertScaleAbs(tempDstNorm, corners);
            assertEquals(100, corners.rows());
            assertEquals(100, corners.cols());
            assertEquals(CvType.CV_8UC1, corners.type());
        }
    }
}
