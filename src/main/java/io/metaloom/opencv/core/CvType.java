package io.metaloom.opencv.core;

public class CvType {

    // Depth constants
    public static final int CV_8U  = 0;
    public static final int CV_8S  = 1;
    public static final int CV_16U = 2;
    public static final int CV_16S = 3;
    public static final int CV_32S = 4;
    public static final int CV_32F = 5;
    public static final int CV_64F = 6;
    public static final int CV_16F = 7;
    public static final int CV_USRTYPE1 = 7;

    // Predefined types
    public static final int CV_8UC1  = CV_8UC(1);
    public static final int CV_8UC2  = CV_8UC(2);
    public static final int CV_8UC3  = CV_8UC(3);
    public static final int CV_8UC4  = CV_8UC(4);
    public static final int CV_8SC1  = CV_8SC(1);
    public static final int CV_8SC2  = CV_8SC(2);
    public static final int CV_8SC3  = CV_8SC(3);
    public static final int CV_8SC4  = CV_8SC(4);
    public static final int CV_16UC1 = CV_16UC(1);
    public static final int CV_16UC2 = CV_16UC(2);
    public static final int CV_16UC3 = CV_16UC(3);
    public static final int CV_16UC4 = CV_16UC(4);
    public static final int CV_16SC1 = CV_16SC(1);
    public static final int CV_16SC2 = CV_16SC(2);
    public static final int CV_16SC3 = CV_16SC(3);
    public static final int CV_16SC4 = CV_16SC(4);
    public static final int CV_32SC1 = CV_32SC(1);
    public static final int CV_32SC2 = CV_32SC(2);
    public static final int CV_32SC3 = CV_32SC(3);
    public static final int CV_32SC4 = CV_32SC(4);
    public static final int CV_32FC1 = CV_32FC(1);
    public static final int CV_32FC2 = CV_32FC(2);
    public static final int CV_32FC3 = CV_32FC(3);
    public static final int CV_32FC4 = CV_32FC(4);
    public static final int CV_64FC1 = CV_64FC(1);
    public static final int CV_64FC2 = CV_64FC(2);
    public static final int CV_64FC3 = CV_64FC(3);
    public static final int CV_64FC4 = CV_64FC(4);

    private static final int CV_CN_MAX = 512;
    private static final int CV_CN_SHIFT = 3;
    private static final int CV_DEPTH_MAX = (1 << CV_CN_SHIFT);

    public static int makeType(int depth, int channels) {
        if (channels <= 0 || channels >= CV_CN_MAX) {
            throw new UnsupportedOperationException(
                    "Channels count should be 1.." + (CV_CN_MAX - 1));
        }
        if (depth < 0 || depth >= CV_DEPTH_MAX) {
            throw new UnsupportedOperationException(
                    "Data type depth should be 0.." + (CV_DEPTH_MAX - 1));
        }
        return (depth & (CV_DEPTH_MAX - 1)) + ((channels - 1) << CV_CN_SHIFT);
    }

    public static int CV_8UC(int ch) { return makeType(CV_8U, ch); }
    public static int CV_8SC(int ch) { return makeType(CV_8S, ch); }
    public static int CV_16UC(int ch) { return makeType(CV_16U, ch); }
    public static int CV_16SC(int ch) { return makeType(CV_16S, ch); }
    public static int CV_32SC(int ch) { return makeType(CV_32S, ch); }
    public static int CV_32FC(int ch) { return makeType(CV_32F, ch); }
    public static int CV_64FC(int ch) { return makeType(CV_64F, ch); }

    public static int channels(int type) {
        return (type >> CV_CN_SHIFT) + 1;
    }

    public static int depth(int type) {
        return type & (CV_DEPTH_MAX - 1);
    }

    public static boolean isInteger(int type) {
        return depth(type) < CV_32F;
    }

    public static int ELEM_SIZE(int type) {
        int d = depth(type);
        int sz;
        switch (d) {
            case CV_8U:
            case CV_8S:
                sz = 1; break;
            case CV_16U:
            case CV_16S:
            case CV_16F:
                sz = 2; break;
            case CV_32S:
            case CV_32F:
                sz = 4; break;
            case CV_64F:
                sz = 8; break;
            default:
                throw new UnsupportedOperationException("Unsupported CvType value: " + type);
        }
        return sz * channels(type);
    }

    public static String typeToString(int type) {
        int d = depth(type);
        int ch = channels(type);
        String depthStr = switch (d) {
            case CV_8U -> "CV_8U";
            case CV_8S -> "CV_8S";
            case CV_16U -> "CV_16U";
            case CV_16S -> "CV_16S";
            case CV_32S -> "CV_32S";
            case CV_32F -> "CV_32F";
            case CV_64F -> "CV_64F";
            case CV_16F -> "CV_16F";
            default -> "CV_USRTYPE1";
        };
        if (ch == 1) {
            return depthStr + "C1";
        }
        return depthStr + "C(" + ch + ")";
    }
}
