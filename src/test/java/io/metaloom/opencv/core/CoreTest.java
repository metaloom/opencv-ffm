package io.metaloom.opencv.core;

import io.metaloom.opencv.OpenCVLoader;
import org.junit.jupiter.api.*;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class CoreTest {

    @BeforeAll
    static void loadLibrary() {
        OpenCVLoader.load(Path.of(System.getProperty("native.lib.dir", "native/build")));
    }

    @Test
    @Order(1)
    void testAdd() {
        try (Mat a = new Mat(2, 2, CvType.CV_32FC1, new Scalar(10));
             Mat b = new Mat(2, 2, CvType.CV_32FC1, new Scalar(20));
             Mat dst = new Mat()) {
            Core.add(a, b, dst);
            double[] val = dst.get(0, 0);
            assertEquals(30.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(2)
    void testSubtract() {
        try (Mat a = new Mat(2, 2, CvType.CV_32FC1, new Scalar(30));
             Mat b = new Mat(2, 2, CvType.CV_32FC1, new Scalar(10));
             Mat dst = new Mat()) {
            Core.subtract(a, b, dst);
            double[] val = dst.get(0, 0);
            assertEquals(20.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(3)
    void testMultiply() {
        try (Mat a = new Mat(2, 2, CvType.CV_32FC1, new Scalar(3));
             Mat b = new Mat(2, 2, CvType.CV_32FC1, new Scalar(4));
             Mat dst = new Mat()) {
            Core.multiply(a, b, dst);
            double[] val = dst.get(0, 0);
            assertEquals(12.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(4)
    void testDivide() {
        try (Mat a = new Mat(2, 2, CvType.CV_32FC1, new Scalar(12));
             Mat b = new Mat(2, 2, CvType.CV_32FC1, new Scalar(4));
             Mat dst = new Mat()) {
            Core.divide(a, b, dst);
            double[] val = dst.get(0, 0);
            assertEquals(3.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(5)
    void testAbsdiff() {
        try (Mat a = new Mat(2, 2, CvType.CV_32FC1, new Scalar(10));
             Mat b = new Mat(2, 2, CvType.CV_32FC1, new Scalar(30));
             Mat dst = new Mat()) {
            Core.absdiff(a, b, dst);
            double[] val = dst.get(0, 0);
            assertEquals(20.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(6)
    void testBitwiseAnd() {
        try (Mat a = new Mat(2, 2, CvType.CV_8UC1, new Scalar(0xFF));
             Mat b = new Mat(2, 2, CvType.CV_8UC1, new Scalar(0x0F));
             Mat dst = new Mat()) {
            Core.bitwise_and(a, b, dst);
            double[] val = dst.get(0, 0);
            assertEquals(0x0F, val[0], 1e-6);
        }
    }

    @Test
    @Order(7)
    void testBitwiseOr() {
        try (Mat a = new Mat(2, 2, CvType.CV_8UC1, new Scalar(0xF0));
             Mat b = new Mat(2, 2, CvType.CV_8UC1, new Scalar(0x0F));
             Mat dst = new Mat()) {
            Core.bitwise_or(a, b, dst);
            double[] val = dst.get(0, 0);
            assertEquals(0xFF, val[0], 1e-6);
        }
    }

    @Test
    @Order(8)
    void testBitwiseNot() {
        try (Mat a = new Mat(2, 2, CvType.CV_8UC1, new Scalar(0));
             Mat dst = new Mat()) {
            Core.bitwise_not(a, dst);
            double[] val = dst.get(0, 0);
            assertEquals(255.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(9)
    void testNorm() {
        try (Mat mat = new Mat(1, 3, CvType.CV_64FC1)) {
            mat.put(0, 0, new double[]{3, 4, 0});
            double l2 = Core.norm(mat, Core.NORM_L2);
            assertEquals(5.0, l2, 1e-6);

            double l1 = Core.norm(mat, Core.NORM_L1);
            assertEquals(7.0, l1, 1e-6);

            double inf = Core.norm(mat, Core.NORM_INF);
            assertEquals(4.0, inf, 1e-6);
        }
    }

    @Test
    @Order(10)
    void testMean() {
        try (Mat mat = new Mat(2, 2, CvType.CV_32FC1, new Scalar(10))) {
            Scalar mean = Core.mean(mat);
            assertEquals(10.0, mean.val[0], 1e-6);
        }
    }

    @Test
    @Order(11)
    void testMinMaxLoc() {
        try (Mat mat = new Mat(3, 3, CvType.CV_32FC1, new Scalar(5))) {
            // Set min and max
            mat.put(1, 2, new float[]{100});
            mat.put(2, 0, new float[]{-50});

            MinMaxLocResult result = Core.minMaxLoc(mat);
            assertEquals(-50.0, result.minVal, 1e-6);
            assertEquals(100.0, result.maxVal, 1e-6);
        }
    }

    @Test
    @Order(12)
    void testCountNonZero() {
        try (Mat mat = new Mat(3, 3, CvType.CV_8UC1, new Scalar(0))) {
            assertEquals(0, Core.countNonZero(mat));
            mat.put(0, 0, new byte[]{1});
            mat.put(1, 1, new byte[]{2});
            assertEquals(2, Core.countNonZero(mat));
        }
    }

    @Test
    @Order(13)
    void testFlip() {
        try (Mat src = new Mat(2, 2, CvType.CV_8UC1);
             Mat dst = new Mat()) {
            src.put(0, 0, new byte[]{1, 2, 3, 4});
            Core.flip(src, dst, 0); // vertical flip
            byte[] result = new byte[4];
            dst.get(0, 0, result);
            assertEquals(3, result[0]);
            assertEquals(4, result[1]);
            assertEquals(1, result[2]);
            assertEquals(2, result[3]);
        }
    }

    @Test
    @Order(14)
    void testTranspose() {
        try (Mat src = new Mat(2, 3, CvType.CV_32FC1);
             Mat dst = new Mat()) {
            Core.transpose(src, dst);
            assertEquals(3, dst.rows());
            assertEquals(2, dst.cols());
        }
    }

    @Test
    @Order(15)
    void testRotate() {
        try (Mat src = new Mat(2, 3, CvType.CV_8UC1, new Scalar(1));
             Mat dst = new Mat()) {
            Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE);
            assertEquals(3, dst.rows());
            assertEquals(2, dst.cols());
        }
    }

    @Test
    @Order(16)
    void testAddWeighted() {
        try (Mat a = new Mat(2, 2, CvType.CV_32FC1, new Scalar(10));
             Mat b = new Mat(2, 2, CvType.CV_32FC1, new Scalar(20));
             Mat dst = new Mat()) {
            Core.addWeighted(a, 0.5, b, 0.5, 0, dst);
            double[] val = dst.get(0, 0);
            assertEquals(15.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(17)
    void testNormalize() {
        try (Mat src = new Mat(1, 4, CvType.CV_32FC1);
             Mat dst = new Mat()) {
            src.put(0, 0, new float[]{1, 2, 3, 4});
            Core.normalize(src, dst, 0, 1, Core.NORM_MINMAX);
            float[] result = new float[4];
            dst.get(0, 0, result);
            assertEquals(0.0f, result[0], 1e-5f);
            assertEquals(1.0f, result[3], 1e-5f);
        }
    }

    @Test
    @Order(18)
    void testSumElems() {
        try (Mat mat = new Mat(2, 2, CvType.CV_32FC1, new Scalar(5))) {
            Scalar sum = Core.sumElems(mat);
            assertEquals(20.0, sum.val[0], 1e-6);
        }
    }

    @Test
    @Order(19)
    void testCompare() {
        try (Mat a = new Mat(2, 2, CvType.CV_32FC1, new Scalar(10));
             Mat b = new Mat(2, 2, CvType.CV_32FC1, new Scalar(20));
             Mat dst = new Mat()) {
            Core.compare(a, b, dst, Core.CMP_LT);
            double[] val = dst.get(0, 0);
            assertEquals(255.0, val[0], 1e-6); // all true
        }
    }

    @Test
    @Order(20)
    void testHconcat() {
        try (Mat a = new Mat(2, 2, CvType.CV_8UC1, new Scalar(1));
             Mat b = new Mat(2, 3, CvType.CV_8UC1, new Scalar(2));
             Mat dst = new Mat()) {
            Core.hconcat(a, b, dst);
            assertEquals(2, dst.rows());
            assertEquals(5, dst.cols());
        }
    }

    @Test
    @Order(21)
    void testVconcat() {
        try (Mat a = new Mat(2, 3, CvType.CV_8UC1, new Scalar(1));
             Mat b = new Mat(3, 3, CvType.CV_8UC1, new Scalar(2));
             Mat dst = new Mat()) {
            Core.vconcat(a, b, dst);
            assertEquals(5, dst.rows());
            assertEquals(3, dst.cols());
        }
    }

    @Test
    @Order(22)
    void testConvertScaleAbs() {
        try (Mat src = new Mat(2, 2, CvType.CV_32FC1, new Scalar(-10));
             Mat dst = new Mat()) {
            Core.convertScaleAbs(src, dst);
            double[] val = dst.get(0, 0);
            assertEquals(10.0, val[0], 1e-6);
        }
    }
}
