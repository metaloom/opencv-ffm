package io.metaloom.opencv;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

/**
 * Central registry of all FFM downcall handles to the native opencv_ffm library.
 * All handles are lazily resolved on first access.
 */
public final class NativeBindings {

    private static final Linker LINKER = Linker.nativeLinker();

    private NativeBindings() {
    }

    private static MethodHandle downcall(String name, FunctionDescriptor desc) {
        return LINKER.downcallHandle(
                OpenCVLoader.lookup().find(name).orElseThrow(
                        () -> new UnsatisfiedLinkError("Symbol not found: " + name)),
                desc);
    }

    // Convenience: pointer type
    static final AddressLayout PTR = ADDRESS;

    // ========================================================================
    // Error handling
    // ========================================================================
    public static final MethodHandle GET_LAST_ERROR = downcall("opencv_get_last_error",
            FunctionDescriptor.of(PTR));

    // ========================================================================
    // Mat lifecycle
    // ========================================================================
    public static final MethodHandle MAT_CREATE = downcall("opencv_mat_create",
            FunctionDescriptor.of(PTR));

    public static final MethodHandle MAT_CREATE_WITH_SIZE = downcall("opencv_mat_create_with_size",
            FunctionDescriptor.of(PTR, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_CREATE_WITH_SCALAR = downcall("opencv_mat_create_with_scalar",
            FunctionDescriptor.of(PTR, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle MAT_CREATE_FROM_DATA = downcall("opencv_mat_create_from_data",
            FunctionDescriptor.of(PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR, JAVA_LONG));

    public static final MethodHandle MAT_DELETE = downcall("opencv_mat_delete",
            FunctionDescriptor.ofVoid(PTR));

    public static final MethodHandle MAT_RELEASE = downcall("opencv_mat_release",
            FunctionDescriptor.ofVoid(PTR));

    public static final MethodHandle MAT_CLONE = downcall("opencv_mat_clone",
            FunctionDescriptor.of(PTR, PTR));

    // ========================================================================
    // Mat metadata
    // ========================================================================
    public static final MethodHandle MAT_ROWS = downcall("opencv_mat_rows",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle MAT_COLS = downcall("opencv_mat_cols",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle MAT_TYPE = downcall("opencv_mat_type",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle MAT_DEPTH = downcall("opencv_mat_depth",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle MAT_CHANNELS = downcall("opencv_mat_channels",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle MAT_DIMS = downcall("opencv_mat_dims",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle MAT_EMPTY = downcall("opencv_mat_empty",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle MAT_TOTAL = downcall("opencv_mat_total",
            FunctionDescriptor.of(JAVA_LONG, PTR));

    public static final MethodHandle MAT_ELEM_SIZE = downcall("opencv_mat_elem_size",
            FunctionDescriptor.of(JAVA_LONG, PTR));

    public static final MethodHandle MAT_ELEM_SIZE1 = downcall("opencv_mat_elem_size1",
            FunctionDescriptor.of(JAVA_LONG, PTR));

    public static final MethodHandle MAT_STEP1 = downcall("opencv_mat_step1",
            FunctionDescriptor.of(JAVA_LONG, PTR, JAVA_INT));

    public static final MethodHandle MAT_DATA_ADDR = downcall("opencv_mat_data_addr",
            FunctionDescriptor.of(PTR, PTR));

    public static final MethodHandle MAT_IS_CONTINUOUS = downcall("opencv_mat_is_continuous",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle MAT_IS_SUBMATRIX = downcall("opencv_mat_is_submatrix",
            FunctionDescriptor.of(JAVA_INT, PTR));

    // ========================================================================
    // Mat data access
    // ========================================================================
    public static final MethodHandle MAT_PUT_B = downcall("opencv_mat_put_b",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_PUT_S = downcall("opencv_mat_put_s",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_PUT_I = downcall("opencv_mat_put_i",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_PUT_F = downcall("opencv_mat_put_f",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_PUT_D = downcall("opencv_mat_put_d",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_GET_B = downcall("opencv_mat_get_b",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_GET_S = downcall("opencv_mat_get_s",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_GET_I = downcall("opencv_mat_get_i",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_GET_F = downcall("opencv_mat_get_f",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_GET_D = downcall("opencv_mat_get_d",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle MAT_GET_PIXEL = downcall("opencv_mat_get_pixel",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, PTR));

    // ========================================================================
    // Mat operations
    // ========================================================================
    public static final MethodHandle MAT_COPY_TO = downcall("opencv_mat_copy_to",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR));

    public static final MethodHandle MAT_COPY_TO_MASKED = downcall("opencv_mat_copy_to_masked",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    public static final MethodHandle MAT_CONVERT_TO = downcall("opencv_mat_convert_to",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle MAT_SET_TO_SCALAR = downcall("opencv_mat_set_to_scalar",
            FunctionDescriptor.of(PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle MAT_SET_TO_MASKED = downcall("opencv_mat_set_to_masked",
            FunctionDescriptor.of(PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, PTR));

    public static final MethodHandle MAT_RESHAPE = downcall("opencv_mat_reshape",
            FunctionDescriptor.of(PTR, PTR, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_ROW = downcall("opencv_mat_row",
            FunctionDescriptor.of(PTR, PTR, JAVA_INT));

    public static final MethodHandle MAT_COL = downcall("opencv_mat_col",
            FunctionDescriptor.of(PTR, PTR, JAVA_INT));

    public static final MethodHandle MAT_ROW_RANGE = downcall("opencv_mat_row_range",
            FunctionDescriptor.of(PTR, PTR, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_COL_RANGE = downcall("opencv_mat_col_range",
            FunctionDescriptor.of(PTR, PTR, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_SUBMAT = downcall("opencv_mat_submat",
            FunctionDescriptor.of(PTR, PTR, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_T = downcall("opencv_mat_t",
            FunctionDescriptor.of(PTR, PTR));

    public static final MethodHandle MAT_INV = downcall("opencv_mat_inv",
            FunctionDescriptor.of(PTR, PTR, JAVA_INT));

    public static final MethodHandle MAT_MUL = downcall("opencv_mat_mul",
            FunctionDescriptor.of(PTR, PTR, PTR, JAVA_DOUBLE));

    public static final MethodHandle MAT_MAT_MUL = downcall("opencv_mat_mat_mul",
            FunctionDescriptor.of(PTR, PTR, PTR));

    public static final MethodHandle MAT_DOT = downcall("opencv_mat_dot",
            FunctionDescriptor.of(JAVA_DOUBLE, PTR, PTR));

    public static final MethodHandle MAT_CROSS = downcall("opencv_mat_cross",
            FunctionDescriptor.of(PTR, PTR, PTR));

    public static final MethodHandle MAT_PUSH_BACK = downcall("opencv_mat_push_back",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR));

    public static final MethodHandle MAT_CREATE_INPLACE = downcall("opencv_mat_create_inplace",
            FunctionDescriptor.ofVoid(PTR, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_CHECK_VECTOR = downcall("opencv_mat_check_vector",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT, JAVA_INT));

    // Mat factories
    public static final MethodHandle MAT_ZEROS = downcall("opencv_mat_zeros",
            FunctionDescriptor.of(PTR, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_ONES = downcall("opencv_mat_ones",
            FunctionDescriptor.of(PTR, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_EYE = downcall("opencv_mat_eye",
            FunctionDescriptor.of(PTR, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle MAT_DIAG = downcall("opencv_mat_diag",
            FunctionDescriptor.of(PTR, PTR, JAVA_INT));

    public static final MethodHandle MAT_DUMP = downcall("opencv_mat_dump",
            FunctionDescriptor.of(PTR, PTR));

    // ========================================================================
    // Core arithmetic
    // ========================================================================
    public static final MethodHandle CORE_ADD = downcall("opencv_core_add",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR, JAVA_INT));

    public static final MethodHandle CORE_SUBTRACT = downcall("opencv_core_subtract",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR, JAVA_INT));

    public static final MethodHandle CORE_MULTIPLY = downcall("opencv_core_multiply",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle CORE_DIVIDE = downcall("opencv_core_divide",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle CORE_ADD_WEIGHTED = downcall("opencv_core_add_weighted",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_DOUBLE, PTR, JAVA_DOUBLE, JAVA_DOUBLE, PTR, JAVA_INT));

    public static final MethodHandle CORE_SCALE_ADD = downcall("opencv_core_scale_add",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_DOUBLE, PTR, PTR));

    public static final MethodHandle CORE_ABS_DIFF = downcall("opencv_core_abs_diff",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    // ========================================================================
    // Core bitwise
    // ========================================================================
    public static final MethodHandle CORE_BITWISE_AND = downcall("opencv_core_bitwise_and",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR));

    public static final MethodHandle CORE_BITWISE_OR = downcall("opencv_core_bitwise_or",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR));

    public static final MethodHandle CORE_BITWISE_XOR = downcall("opencv_core_bitwise_xor",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR));

    public static final MethodHandle CORE_BITWISE_NOT = downcall("opencv_core_bitwise_not",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    // ========================================================================
    // Core comparison
    // ========================================================================
    public static final MethodHandle CORE_COMPARE = downcall("opencv_core_compare",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_INT));

    public static final MethodHandle CORE_MIN = downcall("opencv_core_min",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    public static final MethodHandle CORE_MAX = downcall("opencv_core_max",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    public static final MethodHandle CORE_IN_RANGE = downcall("opencv_core_in_range",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR));

    // ========================================================================
    // Core statistics
    // ========================================================================
    public static final MethodHandle CORE_NORM = downcall("opencv_core_norm",
            FunctionDescriptor.of(JAVA_DOUBLE, PTR, JAVA_INT, PTR));

    public static final MethodHandle CORE_NORM_DIFF = downcall("opencv_core_norm_diff",
            FunctionDescriptor.of(JAVA_DOUBLE, PTR, PTR, JAVA_INT, PTR));

    public static final MethodHandle CORE_MEAN = downcall("opencv_core_mean",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    public static final MethodHandle CORE_MEAN_STD_DEV = downcall("opencv_core_mean_std_dev",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR));

    public static final MethodHandle CORE_MIN_MAX_LOC = downcall("opencv_core_min_max_loc",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR, PTR, PTR));

    public static final MethodHandle CORE_COUNT_NON_ZERO = downcall("opencv_core_count_non_zero",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle CORE_SUM = downcall("opencv_core_sum",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR));

    // ========================================================================
    // Core array operations
    // ========================================================================
    public static final MethodHandle CORE_NORMALIZE = downcall("opencv_core_normalize",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT, PTR));

    public static final MethodHandle CORE_FLIP = downcall("opencv_core_flip",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT));

    public static final MethodHandle CORE_ROTATE = downcall("opencv_core_rotate",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT));

    public static final MethodHandle CORE_TRANSPOSE = downcall("opencv_core_transpose",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR));

    public static final MethodHandle CORE_CONVERT_SCALE_ABS = downcall("opencv_core_convert_scale_abs",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle CORE_LUT = downcall("opencv_core_lut",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    public static final MethodHandle CORE_COPY_MAKE_BORDER = downcall("opencv_core_copy_make_border",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle CORE_HCONCAT = downcall("opencv_core_hconcat",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    public static final MethodHandle CORE_VCONCAT = downcall("opencv_core_vconcat",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR));

    public static final MethodHandle CORE_GEMM = downcall("opencv_core_gemm",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, PTR, JAVA_DOUBLE, PTR, JAVA_INT));

    public static final MethodHandle CORE_DFT = downcall("opencv_core_dft",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_INT));

    public static final MethodHandle CORE_IDFT = downcall("opencv_core_idft",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_INT));

    // ========================================================================
    // Imgproc - Color conversion
    // ========================================================================
    public static final MethodHandle IMGPROC_CVT_COLOR = downcall("opencv_imgproc_cvt_color",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_INT));

    // ========================================================================
    // Imgproc - Resize / Geometric
    // ========================================================================
    public static final MethodHandle IMGPROC_RESIZE = downcall("opencv_imgproc_resize",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_WARP_AFFINE = downcall("opencv_imgproc_warp_affine",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_WARP_PERSPECTIVE = downcall("opencv_imgproc_warp_perspective",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_GET_ROTATION_MATRIX_2D = downcall("opencv_imgproc_get_rotation_matrix_2d",
            FunctionDescriptor.of(PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    // ========================================================================
    // Imgproc - Threshold
    // ========================================================================
    public static final MethodHandle IMGPROC_THRESHOLD = downcall("opencv_imgproc_threshold",
            FunctionDescriptor.of(JAVA_DOUBLE, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_ADAPTIVE_THRESHOLD = downcall("opencv_imgproc_adaptive_threshold",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE));

    // ========================================================================
    // Imgproc - Filtering
    // ========================================================================
    public static final MethodHandle IMGPROC_BLUR = downcall("opencv_imgproc_blur",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_GAUSSIAN_BLUR = downcall("opencv_imgproc_gaussian_blur",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_MEDIAN_BLUR = downcall("opencv_imgproc_median_blur",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT));

    public static final MethodHandle IMGPROC_BILATERAL_FILTER = downcall("opencv_imgproc_bilateral_filter",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_SOBEL = downcall("opencv_imgproc_sobel",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_SCHARR = downcall("opencv_imgproc_scharr",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_LAPLACIAN = downcall("opencv_imgproc_laplacian",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_CANNY = downcall("opencv_imgproc_canny",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT));

    // ========================================================================
    // Imgproc - Morphology
    // ========================================================================
    public static final MethodHandle IMGPROC_ERODE = downcall("opencv_imgproc_erode",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_DILATE = downcall("opencv_imgproc_dilate",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_MORPHOLOGY_EX = downcall("opencv_imgproc_morphology_ex",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_GET_STRUCTURING_ELEMENT = downcall("opencv_imgproc_get_structuring_element",
            FunctionDescriptor.of(PTR, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));

    // ========================================================================
    // Imgproc - Drawing
    // ========================================================================
    public static final MethodHandle IMGPROC_LINE = downcall("opencv_imgproc_line",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle IMGPROC_RECTANGLE = downcall("opencv_imgproc_rectangle",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle IMGPROC_CIRCLE = downcall("opencv_imgproc_circle",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle IMGPROC_ELLIPSE = downcall("opencv_imgproc_ellipse",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT, JAVA_INT));

    public static final MethodHandle IMGPROC_PUT_TEXT = downcall("opencv_imgproc_put_text",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_DOUBLE,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT, JAVA_INT));

    // ========================================================================
    // Imgproc - Contours
    // ========================================================================
    public static final MethodHandle IMGPROC_FIND_CONTOURS = downcall("opencv_imgproc_find_contours",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_INT, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_DRAW_CONTOURS = downcall("opencv_imgproc_draw_contours",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT,
                    JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE,
                    JAVA_INT, JAVA_INT, PTR, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_CONTOUR_AREA = downcall("opencv_imgproc_contour_area",
            FunctionDescriptor.of(JAVA_DOUBLE, PTR, JAVA_INT));

    public static final MethodHandle IMGPROC_ARC_LENGTH = downcall("opencv_imgproc_arc_length",
            FunctionDescriptor.of(JAVA_DOUBLE, PTR, JAVA_INT));

    public static final MethodHandle IMGPROC_BOUNDING_RECT = downcall("opencv_imgproc_bounding_rect",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR, PTR));

    // ========================================================================
    // Imgproc - Histogram
    // ========================================================================
    public static final MethodHandle IMGPROC_EQUALIZE_HIST = downcall("opencv_imgproc_equalize_hist",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR));

    public static final MethodHandle IMGPROC_CALC_HIST_1 = downcall("opencv_imgproc_calc_hist_1",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, PTR, PTR, PTR, JAVA_INT, JAVA_INT));

    public static final MethodHandle IMGPROC_COMPARE_HIST = downcall("opencv_imgproc_compare_hist",
            FunctionDescriptor.of(JAVA_DOUBLE, PTR, PTR, JAVA_INT));

    // ========================================================================
    // Imgproc - Additional feature detection (used by video4j)
    // ========================================================================
    public static final MethodHandle IMGPROC_CORNER_HARRIS = downcall("opencv_imgproc_corner_harris",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_INT, JAVA_DOUBLE, JAVA_INT));

    public static final MethodHandle IMGPROC_GOOD_FEATURES_TO_TRACK = downcall("opencv_imgproc_good_features_to_track",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE, PTR, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_HOUGH_LINES = downcall("opencv_imgproc_hough_lines",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_HOUGH_LINES_P = downcall("opencv_imgproc_hough_lines_p",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE));

    public static final MethodHandle IMGPROC_GET_RECT_SUB_PIX = downcall("opencv_imgproc_get_rect_sub_pix",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, PTR, JAVA_INT));

    // ========================================================================
    // Core - Additional math (used by video4j)
    // ========================================================================
    public static final MethodHandle CORE_SQRT = downcall("opencv_core_sqrt",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR));

    // ========================================================================
    // Video - Optical flow
    // ========================================================================
    public static final MethodHandle VIDEO_CALC_OPTICAL_FLOW_PYR_LK = downcall("opencv_video_calc_optical_flow_pyr_lk",
            FunctionDescriptor.of(JAVA_INT,
                    PTR, PTR, PTR, PTR, PTR, PTR,
                    JAVA_DOUBLE, JAVA_DOUBLE,
                    JAVA_INT,
                    JAVA_INT, JAVA_INT, JAVA_DOUBLE,
                    JAVA_INT,
                    JAVA_DOUBLE));

    // ========================================================================
    // VideoCapture (videoio module)
    // ========================================================================
    public static final MethodHandle VIDEOCAPTURE_CREATE = downcall("opencv_videocapture_create",
            FunctionDescriptor.of(PTR));

    public static final MethodHandle VIDEOCAPTURE_DELETE = downcall("opencv_videocapture_delete",
            FunctionDescriptor.ofVoid(PTR));

    public static final MethodHandle VIDEOCAPTURE_OPEN_FILE = downcall("opencv_videocapture_open_file",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT));

    public static final MethodHandle VIDEOCAPTURE_OPEN_DEVICE = downcall("opencv_videocapture_open_device",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT));

    public static final MethodHandle VIDEOCAPTURE_IS_OPENED = downcall("opencv_videocapture_is_opened",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle VIDEOCAPTURE_RELEASE = downcall("opencv_videocapture_release",
            FunctionDescriptor.ofVoid(PTR));

    public static final MethodHandle VIDEOCAPTURE_READ = downcall("opencv_videocapture_read",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR));

    public static final MethodHandle VIDEOCAPTURE_GRAB = downcall("opencv_videocapture_grab",
            FunctionDescriptor.of(JAVA_INT, PTR));

    public static final MethodHandle VIDEOCAPTURE_RETRIEVE = downcall("opencv_videocapture_retrieve",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT));

    public static final MethodHandle VIDEOCAPTURE_SET = downcall("opencv_videocapture_set",
            FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_DOUBLE));

    public static final MethodHandle VIDEOCAPTURE_GET = downcall("opencv_videocapture_get",
            FunctionDescriptor.of(JAVA_DOUBLE, PTR, JAVA_INT));

    // ========================================================================
    // CascadeClassifier (objdetect module)
    // ========================================================================
    public static final MethodHandle CASCADE_CREATE = downcall("opencv_cascade_create",
            FunctionDescriptor.of(PTR));

    public static final MethodHandle CASCADE_DELETE = downcall("opencv_cascade_delete",
            FunctionDescriptor.ofVoid(PTR));

    public static final MethodHandle CASCADE_LOAD = downcall("opencv_cascade_load",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR));

    public static final MethodHandle CASCADE_DETECT = downcall("opencv_cascade_detect",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, PTR, JAVA_INT));

    public static final MethodHandle CASCADE_DETECT_PARAMS = downcall("opencv_cascade_detect_params",
            FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_DOUBLE, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, PTR, JAVA_INT));
}
