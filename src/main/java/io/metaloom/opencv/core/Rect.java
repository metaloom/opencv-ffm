package io.metaloom.opencv.core;

public class Rect {

    public int x, y, width, height;

    public Rect(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public Rect() {
        this(0, 0, 0, 0);
    }

    public Rect(Point p1, Point p2) {
        x = (int) (p1.x < p2.x ? p1.x : p2.x);
        y = (int) (p1.y < p2.y ? p1.y : p2.y);
        width = (int) (p1.x > p2.x ? p1.x : p2.x) - x;
        height = (int) (p1.y > p2.y ? p1.y : p2.y) - y;
    }

    public Rect(Point p, Size s) {
        x = (int) p.x;
        y = (int) p.y;
        width = (int) s.width;
        height = (int) s.height;
    }

    public Rect(double[] vals) {
        this();
        set(vals);
    }

    public void set(double[] vals) {
        if (vals != null) {
            x = vals.length > 0 ? (int) vals[0] : 0;
            y = vals.length > 1 ? (int) vals[1] : 0;
            width = vals.length > 2 ? (int) vals[2] : 0;
            height = vals.length > 3 ? (int) vals[3] : 0;
        } else {
            x = 0;
            y = 0;
            width = 0;
            height = 0;
        }
    }

    public Rect clone() {
        return new Rect(x, y, width, height);
    }

    public Point tl() {
        return new Point(x, y);
    }

    public Point br() {
        return new Point(x + width, y + height);
    }

    public Size size() {
        return new Size(width, height);
    }

    public double area() {
        return width * height;
    }

    public boolean empty() {
        return width <= 0 || height <= 0;
    }

    public boolean contains(Point p) {
        return x <= p.x && p.x < x + width && y <= p.y && p.y < y + height;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + height;
        result = prime * result + width;
        result = prime * result + x;
        result = prime * result + y;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Rect other)) return false;
        return x == other.x && y == other.y && width == other.width && height == other.height;
    }

    @Override
    public String toString() {
        return "{" + x + ", " + y + ", " + width + "x" + height + "}";
    }
}
