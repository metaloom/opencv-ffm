package io.metaloom.opencv.core;

public class MatOfInt extends Mat {

    public MatOfInt() {
        super();
    }

    public MatOfInt(int... values) {
        super();
        fromArray(values);
    }

    public void fromArray(int... values) {
        if (values == null || values.length == 0) {
            return;
        }
        create(1, values.length, CvType.CV_32SC1);
        put(0, 0, values);
    }

    public int[] toArray() {
        int count = (int) total();
        if (count <= 0) {
            return new int[0];
        }
        int[] data = new int[count];
        get(0, 0, data);
        return data;
    }
}
