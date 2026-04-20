package io.metaloom.opencv.core;

public class Range {

    public int start, end;

    public Range(int start, int end) {
        this.start = start;
        this.end = end;
    }

    public Range() {
        this(0, 0);
    }

    public Range(double[] vals) {
        this();
        set(vals);
    }

    public void set(double[] vals) {
        if (vals != null) {
            start = vals.length > 0 ? (int) vals[0] : 0;
            end = vals.length > 1 ? (int) vals[1] : 0;
        } else {
            start = 0;
            end = 0;
        }
    }

    public int size() {
        return empty() ? 0 : end - start;
    }

    public boolean empty() {
        return end <= start;
    }

    public static Range all() {
        return new Range(Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    public Range clone() {
        return new Range(start, end);
    }

    public Range intersection(Range r) {
        int s = Math.max(start, r.start);
        int e = Math.min(end, r.end);
        return new Range(s, e);
    }

    public Range shift(int delta) {
        return new Range(start + delta, end + delta);
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + end;
        result = prime * result + start;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Range other)) return false;
        return start == other.start && end == other.end;
    }

    @Override
    public String toString() {
        return "[" + start + ", " + end + ")";
    }
}
