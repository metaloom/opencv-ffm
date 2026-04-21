package io.metaloom.opencv.core;

import java.util.Arrays;
import java.util.List;

public class MatOfPoint extends Mat {

    public MatOfPoint() {
        super();
    }

    public MatOfPoint(Point... points) {
        super();
        fromArray(points);
    }

    public void fromArray(Point... points) {
        if (points == null || points.length == 0) {
            return;
        }
        create(points.length, 1, CvType.CV_32FC2);
        float[] data = new float[points.length * 2];
        for (int i = 0; i < points.length; i++) {
            data[i * 2] = (float) points[i].x;
            data[i * 2 + 1] = (float) points[i].y;
        }
        put(0, 0, data);
    }

    public Point[] toArray() {
        int pointCount = checkVector(2, -1, true);
        if (pointCount <= 0) {
            return new Point[0];
        }

        Point[] points = new Point[pointCount];
        int depth = CvType.depth(type());
        if (depth == CvType.CV_32S) {
            int[] data = new int[pointCount * 2];
            get(0, 0, data);
            for (int i = 0; i < pointCount; i++) {
                points[i] = new Point(data[i * 2], data[i * 2 + 1]);
            }
        } else if (depth == CvType.CV_64F) {
            double[] data = new double[pointCount * 2];
            get(0, 0, data);
            for (int i = 0; i < pointCount; i++) {
                points[i] = new Point(data[i * 2], data[i * 2 + 1]);
            }
        } else {
            float[] data = new float[pointCount * 2];
            get(0, 0, data);
            for (int i = 0; i < pointCount; i++) {
                points[i] = new Point(data[i * 2], data[i * 2 + 1]);
            }
        }
        return points;
    }

    public List<Point> toList() {
        return Arrays.asList(toArray());
    }
}
