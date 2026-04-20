package io.metaloom.opencv.core;

public class MinMaxLocResult {

    public double minVal;
    public double maxVal;
    public Point minLoc;
    public Point maxLoc;

    public MinMaxLocResult() {
        minVal = 0;
        maxVal = 0;
        minLoc = new Point();
        maxLoc = new Point();
    }

    public MinMaxLocResult(double minVal, double maxVal, Point minLoc, Point maxLoc) {
        this.minVal = minVal;
        this.maxVal = maxVal;
        this.minLoc = minLoc;
        this.maxLoc = maxLoc;
    }
}
