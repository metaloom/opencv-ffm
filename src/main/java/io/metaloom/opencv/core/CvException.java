package io.metaloom.opencv.core;

public class CvException extends RuntimeException {

    public CvException(String msg) {
        super(msg);
    }

    @Override
    public String toString() {
        return "CvException [" + getMessage() + "]";
    }
}
