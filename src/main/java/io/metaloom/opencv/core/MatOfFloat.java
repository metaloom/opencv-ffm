package io.metaloom.opencv.core;

public class MatOfFloat extends Mat {

    public MatOfFloat() {
        super();
    }

    public MatOfFloat(float... values) {
        super();
        fromArray(values);
    }

    public void fromArray(float... values) {
        if (values == null || values.length == 0) {
            return;
        }
        create(1, values.length, CvType.CV_32FC1);
        put(0, 0, values);
    }

    public float[] toArray() {
        int count = (int) total();
        if (count <= 0) {
            return new float[0];
        }
        float[] data = new float[count];
        get(0, 0, data);
        return data;
    }
}
