package io.metaloom.opencv.core;

import io.metaloom.opencv.OpenCVLoader;
import org.junit.jupiter.api.*;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class MatTest {

    @BeforeAll
    static void loadLibrary() {
        OpenCVLoader.load(Path.of(System.getProperty("native.lib.dir", "native/build")));
    }

    @Test
    @Order(1)
    void testCreateEmpty() {
        try (Mat mat = new Mat()) {
            assertTrue(mat.empty());
            assertEquals(0, mat.rows());
            assertEquals(0, mat.cols());
        }
    }

    @Test
    @Order(2)
    void testCreateWithSize() {
        try (Mat mat = new Mat(3, 4, CvType.CV_8UC1)) {
            assertFalse(mat.empty());
            assertEquals(3, mat.rows());
            assertEquals(4, mat.cols());
            assertEquals(CvType.CV_8UC1, mat.type());
            assertEquals(1, mat.channels());
            assertEquals(CvType.CV_8U, mat.depth());
        }
    }

    @Test
    @Order(3)
    void testCreateWithScalar() {
        try (Mat mat = new Mat(2, 2, CvType.CV_8UC3, new Scalar(10, 20, 30))) {
            assertFalse(mat.empty());
            assertEquals(2, mat.rows());
            assertEquals(2, mat.cols());
            assertEquals(CvType.CV_8UC3, mat.type());
            assertEquals(3, mat.channels());

            double[] pixel = mat.get(0, 0);
            assertNotNull(pixel);
            assertEquals(3, pixel.length);
            assertEquals(10.0, pixel[0], 1e-6);
            assertEquals(20.0, pixel[1], 1e-6);
            assertEquals(30.0, pixel[2], 1e-6);
        }
    }

    @Test
    @Order(4)
    void testZerosOnesEye() {
        try (Mat zeros = Mat.zeros(3, 3, CvType.CV_32FC1);
             Mat ones = Mat.ones(3, 3, CvType.CV_32FC1);
             Mat eye = Mat.eye(3, 3, CvType.CV_32FC1)) {

            assertEquals(3, zeros.rows());
            double[] zVal = zeros.get(0, 0);
            assertEquals(0.0, zVal[0], 1e-6);

            double[] oVal = ones.get(1, 1);
            assertEquals(1.0, oVal[0], 1e-6);

            // Eye: diagonal element
            double[] eVal = eye.get(1, 1);
            assertEquals(1.0, eVal[0], 1e-6);
            // Eye: off-diagonal element
            double[] offVal = eye.get(0, 1);
            assertEquals(0.0, offVal[0], 1e-6);
        }
    }

    @Test
    @Order(5)
    void testClone() {
        try (Mat src = new Mat(2, 2, CvType.CV_8UC1, new Scalar(42));
             Mat clone = src.clone()) {
            assertEquals(src.rows(), clone.rows());
            assertEquals(src.cols(), clone.cols());
            assertEquals(src.type(), clone.type());
            double[] srcPx = src.get(0, 0);
            double[] clnPx = clone.get(0, 0);
            assertEquals(srcPx[0], clnPx[0], 1e-6);
        }
    }

    @Test
    @Order(6)
    void testPutGetBytes() {
        try (Mat mat = new Mat(2, 2, CvType.CV_8UC1)) {
            byte[] data = {1, 2, 3, 4};
            int written = mat.put(0, 0, data);
            assertEquals(4, written);

            byte[] readBack = new byte[4];
            int read = mat.get(0, 0, readBack);
            assertEquals(4, read);
            assertArrayEquals(data, readBack);
        }
    }

    @Test
    @Order(7)
    void testPutGetFloats() {
        try (Mat mat = new Mat(2, 2, CvType.CV_32FC1)) {
            float[] data = {1.5f, 2.5f, 3.5f, 4.5f};
            mat.put(0, 0, data);

            float[] readBack = new float[4];
            mat.get(0, 0, readBack);
            assertArrayEquals(data, readBack, 1e-6f);
        }
    }

    @Test
    @Order(8)
    void testPutGetDoubles() {
        try (Mat mat = new Mat(2, 2, CvType.CV_64FC1)) {
            double[] data = {1.1, 2.2, 3.3, 4.4};
            mat.put(0, 0, data);

            double[] readBack = new double[4];
            mat.get(0, 0, readBack);
            assertArrayEquals(data, readBack, 1e-10);
        }
    }

    @Test
    @Order(9)
    void testSubmat() {
        try (Mat src = new Mat(4, 4, CvType.CV_8UC1, new Scalar(5));
             Mat sub = src.submat(1, 3, 1, 3)) {
            assertEquals(2, sub.rows());
            assertEquals(2, sub.cols());
            assertTrue(sub.isSubmatrix());
            double[] val = sub.get(0, 0);
            assertEquals(5.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(10)
    void testRowCol() {
        try (Mat src = new Mat(3, 3, CvType.CV_32FC1, new Scalar(0))) {
            float[] rowData = {1, 2, 3};
            src.put(1, 0, rowData);

            try (Mat row = src.row(1)) {
                assertEquals(1, row.rows());
                assertEquals(3, row.cols());
                float[] out = new float[3];
                row.get(0, 0, out);
                assertArrayEquals(rowData, out, 1e-6f);
            }
        }
    }

    @Test
    @Order(11)
    void testCopyTo() {
        try (Mat src = new Mat(2, 2, CvType.CV_8UC1, new Scalar(99));
             Mat dst = new Mat()) {
            src.copyTo(dst);
            assertEquals(2, dst.rows());
            assertEquals(2, dst.cols());
            double[] val = dst.get(0, 0);
            assertEquals(99.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(12)
    void testConvertTo() {
        try (Mat src = new Mat(2, 2, CvType.CV_8UC1, new Scalar(100));
             Mat dst = new Mat()) {
            src.convertTo(dst, CvType.CV_32FC1);
            assertEquals(CvType.CV_32FC1, dst.type());
            double[] val = dst.get(0, 0);
            assertEquals(100.0, val[0], 1e-6);
        }
    }

    @Test
    @Order(13)
    void testSetTo() {
        try (Mat mat = new Mat(2, 2, CvType.CV_8UC3)) {
            mat.setTo(new Scalar(255, 128, 64));
            double[] val = mat.get(0, 0);
            assertEquals(255.0, val[0], 1e-6);
            assertEquals(128.0, val[1], 1e-6);
            assertEquals(64.0, val[2], 1e-6);
        }
    }

    @Test
    @Order(14)
    void testReshape() {
        try (Mat src = new Mat(2, 6, CvType.CV_8UC1, new Scalar(7));
             Mat reshaped = src.reshape(1, 3)) {
            assertEquals(3, reshaped.rows());
            assertEquals(4, reshaped.cols());
        }
    }

    @Test
    @Order(15)
    void testTranspose() {
        try (Mat src = new Mat(2, 3, CvType.CV_32FC1);
             Mat t = src.t()) {
            assertEquals(3, t.rows());
            assertEquals(2, t.cols());
        }
    }

    @Test
    @Order(16)
    void testDot() {
        try (Mat a = new Mat(1, 3, CvType.CV_64FC1);
             Mat b = new Mat(1, 3, CvType.CV_64FC1)) {
            a.put(0, 0, new double[]{1, 2, 3});
            b.put(0, 0, new double[]{4, 5, 6});
            double dot = a.dot(b);
            assertEquals(32.0, dot, 1e-6); // 1*4 + 2*5 + 3*6
        }
    }

    @Test
    @Order(17)
    void testMetadata() {
        try (Mat mat = new Mat(10, 20, CvType.CV_8UC3)) {
            assertEquals(10, mat.rows());
            assertEquals(20, mat.cols());
            assertEquals(3, mat.channels());
            assertEquals(CvType.CV_8U, mat.depth());
            assertEquals(200, mat.total()); // 10 * 20
            assertEquals(3, mat.elemSize()); // 3 channels * 1 byte
            assertEquals(1, mat.elemSize1()); // 1 byte per channel
            assertEquals(2, mat.dims());
            assertTrue(mat.isContinuous());
            assertFalse(mat.isSubmatrix());
            assertEquals(20, mat.width());
            assertEquals(10, mat.height());
        }
    }

    @Test
    @Order(18)
    void testDump() {
        try (Mat mat = new Mat(2, 2, CvType.CV_8UC1, new Scalar(42))) {
            String dump = mat.dump();
            assertNotNull(dump);
            assertTrue(dump.contains("42"));
        }
    }

    @Test
    @Order(19)
    void testMatMul() {
        try (Mat a = new Mat(2, 3, CvType.CV_64FC1);
             Mat b = new Mat(3, 2, CvType.CV_64FC1)) {
            a.put(0, 0, new double[]{1, 2, 3, 4, 5, 6});
            b.put(0, 0, new double[]{7, 8, 9, 10, 11, 12});
            try (Mat c = a.matMul(b)) {
                assertEquals(2, c.rows());
                assertEquals(2, c.cols());
                double[] val = c.get(0, 0);
                assertEquals(58.0, val[0], 1e-6); // 1*7 + 2*9 + 3*11
            }
        }
    }

    @Test
    @Order(20)
    void testAutoCloseable() {
        Mat mat = new Mat(2, 2, CvType.CV_8UC1);
        assertFalse(mat.empty());
        mat.close();
        assertThrows(CvException.class, mat::rows);
    }

    @Test
    @Order(21)
    void testToString() {
        try (Mat mat = new Mat(3, 4, CvType.CV_8UC3)) {
            String s = mat.toString();
            assertTrue(s.contains("3*4"));
            assertTrue(s.contains("CV_8UC"));
        }
    }
}
