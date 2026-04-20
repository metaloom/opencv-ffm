package io.metaloom.opencv.core;

public class Scalar {

    public double[] val;

    public Scalar(double v0, double v1, double v2, double v3) {
        val = new double[]{v0, v1, v2, v3};
    }

    public Scalar(double v0, double v1, double v2) {
        val = new double[]{v0, v1, v2, 0};
    }

    public Scalar(double v0, double v1) {
        val = new double[]{v0, v1, 0, 0};
    }

    public Scalar(double v0) {
        val = new double[]{v0, 0, 0, 0};
    }

    public Scalar(double[] vals) {
        if (vals != null && vals.length == 4) {
            val = vals.clone();
        } else {
            val = new double[4];
            set(vals);
        }
    }

    public void set(double[] vals) {
        if (vals != null) {
            for (int i = 0; i < Math.min(vals.length, 4); i++) {
                val[i] = vals[i];
            }
            for (int i = vals.length; i < 4; i++) {
                val[i] = 0;
            }
        } else {
            val[0] = val[1] = val[2] = val[3] = 0;
        }
    }

    public static Scalar all(double v) {
        return new Scalar(v, v, v, v);
    }

    public Scalar clone() {
        return new Scalar(val);
    }

    public Scalar mul(Scalar it, double scale) {
        return new Scalar(val[0] * it.val[0] * scale, val[1] * it.val[1] * scale,
                val[2] * it.val[2] * scale, val[3] * it.val[3] * scale);
    }

    public Scalar mul(Scalar it) {
        return mul(it, 1);
    }

    public Scalar conj() {
        return new Scalar(val[0], -val[1], -val[2], -val[3]);
    }

    public boolean isReal() {
        return val[1] == 0 && val[2] == 0 && val[3] == 0;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + java.util.Arrays.hashCode(val);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Scalar other)) return false;
        return java.util.Arrays.equals(val, other.val);
    }

    @Override
    public String toString() {
        return "[" + val[0] + ", " + val[1] + ", " + val[2] + ", " + val[3] + "]";
    }
}
