package io.metaloom.opencv.imgproc;

import io.metaloom.opencv.OpenCVLoader;
import io.metaloom.opencv.core.*;
import org.junit.jupiter.api.*;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ImgprocTest {

    @BeforeAll
    static void loadLibrary() {
        OpenCVLoader.load(Path.of(System.getProperty("native.lib.dir", "native/build")));
    }

    @Test
    @Order(1)
    void testCvtColor() {
        try (Mat bgr = new Mat(4, 4, CvType.CV_8UC3, new Scalar(100, 150, 200));
             Mat gray = new Mat()) {
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
            assertEquals(4, gray.rows());
            assertEquals(4, gray.cols());
            assertEquals(CvType.CV_8UC1, gray.type());
            // Verify a pixel value exists
            double[] val = gray.get(0, 0);
            assertNotNull(val);
            assertTrue(val[0] > 0);
        }
    }

    @Test
    @Order(2)
    void testResize() {
        try (Mat src = new Mat(100, 100, CvType.CV_8UC3, new Scalar(128, 128, 128));
             Mat dst = new Mat()) {
            Imgproc.resize(src, dst, new Size(50, 50));
            assertEquals(50, dst.rows());
            assertEquals(50, dst.cols());
        }
    }

    @Test
    @Order(3)
    void testResizeWithFactor() {
        try (Mat src = new Mat(100, 200, CvType.CV_8UC1, new Scalar(0));
             Mat dst = new Mat()) {
            Imgproc.resize(src, dst, new Size(0, 0), 0.5, 0.5, Imgproc.INTER_LINEAR);
            assertEquals(50, dst.rows());
            assertEquals(100, dst.cols());
        }
    }

    @Test
    @Order(4)
    void testThreshold() {
        try (Mat src = new Mat(4, 4, CvType.CV_8UC1, new Scalar(100));
             Mat dst = new Mat()) {
            // Set some pixels above threshold
            src.put(0, 0, new byte[]{(byte) 200});
            src.put(1, 1, new byte[]{(byte) 200});

            double thresh = Imgproc.threshold(src, dst, 128, 255, Imgproc.THRESH_BINARY);
            assertEquals(128.0, thresh, 1e-6);
            double[] val00 = dst.get(0, 0);
            assertEquals(255.0, val00[0], 1e-6); // above threshold
            double[] val01 = dst.get(0, 1);
            assertEquals(0.0, val01[0], 1e-6); // below threshold
        }
    }

    @Test
    @Order(5)
    void testGaussianBlur() {
        try (Mat src = new Mat(10, 10, CvType.CV_8UC1, new Scalar(128));
             Mat dst = new Mat()) {
            Imgproc.GaussianBlur(src, dst, new Size(3, 3), 1.0);
            assertEquals(10, dst.rows());
            assertEquals(10, dst.cols());
            // Check center pixel hasn't changed much (uniform image)
            double[] val = dst.get(5, 5);
            assertEquals(128.0, val[0], 2.0);
        }
    }

    @Test
    @Order(6)
    void testBlur() {
        try (Mat src = new Mat(10, 10, CvType.CV_8UC1, new Scalar(100));
             Mat dst = new Mat()) {
            Imgproc.blur(src, dst, new Size(3, 3));
            assertEquals(10, dst.rows());
            assertEquals(10, dst.cols());
        }
    }

    @Test
    @Order(7)
    void testMedianBlur() {
        try (Mat src = new Mat(10, 10, CvType.CV_8UC1, new Scalar(100));
             Mat dst = new Mat()) {
            Imgproc.medianBlur(src, dst, 3);
            assertEquals(10, dst.rows());
            assertEquals(10, dst.cols());
        }
    }

    @Test
    @Order(8)
    void testCanny() {
        try (Mat src = new Mat(20, 20, CvType.CV_8UC1, new Scalar(0));
             Mat edges = new Mat()) {
            // Draw a bright rectangle in the center
            for (int r = 5; r < 15; r++) {
                for (int c = 5; c < 15; c++) {
                    src.put(r, c, new byte[]{(byte) 255});
                }
            }
            Imgproc.Canny(src, edges, 50, 150);
            assertEquals(20, edges.rows());
            assertEquals(20, edges.cols());
            assertEquals(CvType.CV_8UC1, edges.type());
        }
    }

    @Test
    @Order(9)
    void testErodeDilate() {
        try (Mat src = new Mat(10, 10, CvType.CV_8UC1, new Scalar(255));
             Mat eroded = new Mat();
             Mat dilated = new Mat();
             Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3))) {
            Imgproc.erode(src, eroded, kernel);
            assertEquals(10, eroded.rows());
            assertEquals(10, eroded.cols());

            Imgproc.dilate(src, dilated, kernel);
            assertEquals(10, dilated.rows());
            assertEquals(10, dilated.cols());
        }
    }

    @Test
    @Order(10)
    void testGetStructuringElement() {
        try (Mat rect = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
             Mat ellipse = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
             Mat cross = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(5, 5))) {
            assertEquals(5, rect.rows());
            assertEquals(5, rect.cols());
            assertEquals(5, ellipse.rows());
            assertEquals(5, cross.rows());
        }
    }

    @Test
    @Order(11)
    void testMorphologyEx() {
        try (Mat src = new Mat(20, 20, CvType.CV_8UC1, new Scalar(128));
             Mat dst = new Mat();
             Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3))) {
            Imgproc.morphologyEx(src, dst, Imgproc.MORPH_OPEN, kernel);
            assertEquals(20, dst.rows());
        }
    }

    @Test
    @Order(12)
    void testDrawLine() {
        try (Mat img = new Mat(100, 100, CvType.CV_8UC3, new Scalar(0, 0, 0))) {
            Imgproc.line(img, new Point(10, 10), new Point(90, 90), new Scalar(255, 0, 0), 2);
            // Check that the line was drawn (pixel at center should be non-zero)
            double[] val = img.get(50, 50);
            // At least one channel should be non-zero along the line
            assertTrue(val[0] > 0 || val[1] > 0 || val[2] > 0);
        }
    }

    @Test
    @Order(13)
    void testDrawRectangle() {
        try (Mat img = new Mat(100, 100, CvType.CV_8UC3, new Scalar(0, 0, 0))) {
            Imgproc.rectangle(img, new Point(10, 10), new Point(90, 90), new Scalar(0, 255, 0), 2);
            // Check a pixel on the top border
            double[] val = img.get(10, 50);
            assertTrue(val[1] > 0); // green channel
        }
    }

    @Test
    @Order(14)
    void testDrawCircle() {
        try (Mat img = new Mat(100, 100, CvType.CV_8UC3, new Scalar(0, 0, 0))) {
            Imgproc.circle(img, new Point(50, 50), 30, new Scalar(0, 0, 255), 2);
            // Check pixel at top of circle
            double[] val = img.get(20, 50);
            assertTrue(val[2] > 0); // red channel
        }
    }

    @Test
    @Order(15)
    void testPutText() {
        try (Mat img = new Mat(100, 200, CvType.CV_8UC3, new Scalar(0, 0, 0))) {
            Imgproc.putText(img, "Hello", new Point(10, 50),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            // Check that some pixels were drawn (text area should have non-zero values)
            int nonZero = 0;
            for (int r = 30; r < 60; r++) {
                for (int c = 10; c < 100; c++) {
                    double[] px = img.get(r, c);
                    if (px[0] > 0) nonZero++;
                }
            }
            assertTrue(nonZero > 0, "Text should have been drawn");
        }
    }

    @Test
    @Order(16)
    void testEqualizeHist() {
        try (Mat src = new Mat(4, 4, CvType.CV_8UC1);
             Mat dst = new Mat()) {
            byte[] data = new byte[16];
            for (int i = 0; i < 16; i++) {
                data[i] = (byte) (i * 16);
            }
            src.put(0, 0, data);
            Imgproc.equalizeHist(src, dst);
            assertEquals(4, dst.rows());
            assertEquals(4, dst.cols());
            assertEquals(CvType.CV_8UC1, dst.type());
        }
    }

    @Test
    @Order(17)
    void testRectangleFromRect() {
        try (Mat img = new Mat(100, 100, CvType.CV_8UC3, new Scalar(0, 0, 0))) {
            Imgproc.rectangle(img, new Rect(20, 20, 60, 60), new Scalar(255, 255, 0), 3);
            double[] val = img.get(20, 50);
            assertTrue(val[0] > 0 || val[1] > 0);
        }
    }

    @Test
    @Order(18)
    void testSobel() {
        try (Mat src = new Mat(10, 10, CvType.CV_8UC1, new Scalar(0));
             Mat dst = new Mat()) {
            // Create horizontal edge
            for (int c = 0; c < 10; c++) {
                for (int r = 5; r < 10; r++) {
                    src.put(r, c, new byte[]{(byte) 255});
                }
            }
            Imgproc.Sobel(src, dst, CvType.CV_16S, 0, 1);
            assertEquals(10, dst.rows());
            assertEquals(CvType.CV_16SC1, dst.type());
        }
    }

    @Test
    @Order(19)
    void testWarpAffine() {
        try (Mat src = new Mat(100, 100, CvType.CV_8UC1, new Scalar(128));
             Mat M = Imgproc.getRotationMatrix2D(new Point(50, 50), 45, 1.0);
             Mat dst = new Mat()) {
            Imgproc.warpAffine(src, dst, M, new Size(100, 100));
            assertEquals(100, dst.rows());
            assertEquals(100, dst.cols());
        }
    }

    @Test
    @Order(20)
    void testAdaptiveThreshold() {
        try (Mat src = new Mat(10, 10, CvType.CV_8UC1, new Scalar(100));
             Mat dst = new Mat()) {
            // Must have odd blockSize >= 3
            Imgproc.adaptiveThreshold(src, dst, 255,
                    Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 3, 5);
            assertEquals(10, dst.rows());
            assertEquals(CvType.CV_8UC1, dst.type());
        }
    }
}
