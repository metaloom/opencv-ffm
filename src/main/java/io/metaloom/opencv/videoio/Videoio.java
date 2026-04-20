package io.metaloom.opencv.videoio;

/**
 * VideoIO property constants, matching OpenCV's cv::VideoCaptureProperties.
 */
public final class Videoio {

    private Videoio() {
    }

    // Capture API backends
    public static final int CAP_ANY         = 0;
    public static final int CAP_V4L         = 200;
    public static final int CAP_V4L2        = CAP_V4L;
    public static final int CAP_FFMPEG      = 1900;
    public static final int CAP_GSTREAMER   = 1800;

    // Capture properties
    public static final int CAP_PROP_POS_MSEC       = 0;
    public static final int CAP_PROP_POS_FRAMES     = 1;
    public static final int CAP_PROP_POS_AVI_RATIO  = 2;
    public static final int CAP_PROP_FRAME_WIDTH    = 3;
    public static final int CAP_PROP_FRAME_HEIGHT   = 4;
    public static final int CAP_PROP_FPS            = 5;
    public static final int CAP_PROP_FOURCC         = 6;
    public static final int CAP_PROP_FRAME_COUNT    = 7;
    public static final int CAP_PROP_FORMAT         = 8;
    public static final int CAP_PROP_MODE           = 9;
    public static final int CAP_PROP_BRIGHTNESS     = 10;
    public static final int CAP_PROP_CONTRAST       = 11;
    public static final int CAP_PROP_SATURATION     = 12;
    public static final int CAP_PROP_HUE            = 13;
    public static final int CAP_PROP_GAIN           = 14;
    public static final int CAP_PROP_EXPOSURE       = 15;
    public static final int CAP_PROP_CONVERT_RGB    = 16;
    public static final int CAP_PROP_RECTIFICATION  = 18;
    public static final int CAP_PROP_MONOCHROME     = 19;
    public static final int CAP_PROP_SHARPNESS      = 20;
    public static final int CAP_PROP_AUTO_EXPOSURE  = 21;
    public static final int CAP_PROP_GAMMA          = 22;
    public static final int CAP_PROP_TEMPERATURE    = 23;
    public static final int CAP_PROP_TRIGGER        = 24;
    public static final int CAP_PROP_TRIGGER_DELAY  = 25;
    public static final int CAP_PROP_WHITE_BALANCE_BLUE_U  = 17;
    public static final int CAP_PROP_WHITE_BALANCE_RED_V   = 26;
    public static final int CAP_PROP_ZOOM           = 27;
    public static final int CAP_PROP_FOCUS          = 28;
    public static final int CAP_PROP_GUID           = 29;
    public static final int CAP_PROP_ISO_SPEED      = 30;
    public static final int CAP_PROP_BACKLIGHT      = 32;
    public static final int CAP_PROP_PAN            = 33;
    public static final int CAP_PROP_TILT           = 34;
    public static final int CAP_PROP_ROLL           = 35;
    public static final int CAP_PROP_IRIS           = 36;
    public static final int CAP_PROP_SETTINGS       = 37;
    public static final int CAP_PROP_BUFFERSIZE     = 38;
    public static final int CAP_PROP_AUTOFOCUS      = 39;

    /**
     * Compute a FOURCC code from four characters.
     */
    public static int fourcc(char c1, char c2, char c3, char c4) {
        return (c1 & 0xFF) | ((c2 & 0xFF) << 8) | ((c3 & 0xFF) << 16) | ((c4 & 0xFF) << 24);
    }
}
