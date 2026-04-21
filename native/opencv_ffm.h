#ifndef OPENCV_FFM_H
#define OPENCV_FFM_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Error handling
 * ============================================================ */
const char* opencv_get_last_error(void);

/* ============================================================
 * Mat lifecycle
 * ============================================================ */
void* opencv_mat_create(void);
void* opencv_mat_create_with_size(int rows, int cols, int type);
void* opencv_mat_create_with_scalar(int rows, int cols, int type, double s0, double s1, double s2, double s3);
void* opencv_mat_create_from_data(int rows, int cols, int type, void* data, long step);
void  opencv_mat_delete(void* mat);
void  opencv_mat_release(void* mat);
void* opencv_mat_clone(void* mat);

/* ============================================================
 * Mat metadata
 * ============================================================ */
int    opencv_mat_rows(void* mat);
int    opencv_mat_cols(void* mat);
int    opencv_mat_type(void* mat);
int    opencv_mat_depth(void* mat);
int    opencv_mat_channels(void* mat);
int    opencv_mat_dims(void* mat);
int    opencv_mat_empty(void* mat);
long   opencv_mat_total(void* mat);
long   opencv_mat_elem_size(void* mat);
long   opencv_mat_elem_size1(void* mat);
long   opencv_mat_step1(void* mat, int i);
void*  opencv_mat_data_addr(void* mat);
int    opencv_mat_is_continuous(void* mat);
int    opencv_mat_is_submatrix(void* mat);

/* ============================================================
 * Mat data access
 * ============================================================ */
int opencv_mat_put_b(void* mat, int row, int col, int count, const char* data);
int opencv_mat_put_s(void* mat, int row, int col, int count, const short* data);
int opencv_mat_put_i(void* mat, int row, int col, int count, const int* data);
int opencv_mat_put_f(void* mat, int row, int col, int count, const float* data);
int opencv_mat_put_d(void* mat, int row, int col, int count, const double* data);

int opencv_mat_get_b(void* mat, int row, int col, int count, char* data);
int opencv_mat_get_s(void* mat, int row, int col, int count, short* data);
int opencv_mat_get_i(void* mat, int row, int col, int count, int* data);
int opencv_mat_get_f(void* mat, int row, int col, int count, float* data);
int opencv_mat_get_d(void* mat, int row, int col, int count, double* data);

/* Get a single pixel as doubles (up to 4 channels) - returns number of channels */
int opencv_mat_get_pixel(void* mat, int row, int col, double* values);

/* ============================================================
 * Mat operations
 * ============================================================ */
int   opencv_mat_copy_to(void* src, void* dst);
int   opencv_mat_copy_to_masked(void* src, void* dst, void* mask);
int   opencv_mat_convert_to(void* src, void* dst, int rtype, double alpha, double beta);
void* opencv_mat_set_to_scalar(void* mat, double s0, double s1, double s2, double s3);
void* opencv_mat_set_to_masked(void* mat, double s0, double s1, double s2, double s3, void* mask);
void* opencv_mat_reshape(void* mat, int cn, int rows);
void* opencv_mat_row(void* mat, int y);
void* opencv_mat_col(void* mat, int x);
void* opencv_mat_row_range(void* mat, int startrow, int endrow);
void* opencv_mat_col_range(void* mat, int startcol, int endcol);
void* opencv_mat_submat(void* mat, int rowStart, int rowEnd, int colStart, int colEnd);
void* opencv_mat_t(void* mat);
void* opencv_mat_inv(void* mat, int method);
void* opencv_mat_mul(void* mat, void* other, double scale);
void* opencv_mat_mat_mul(void* mat, void* other);
double opencv_mat_dot(void* mat, void* other);
void* opencv_mat_cross(void* mat, void* other);
int   opencv_mat_push_back(void* mat, void* other);
void  opencv_mat_create_inplace(void* mat, int rows, int cols, int type);
int   opencv_mat_check_vector(void* mat, int elemChannels, int depth, int requireContinuous);

/* Mat factories */
void* opencv_mat_zeros(int rows, int cols, int type);
void* opencv_mat_ones(int rows, int cols, int type);
void* opencv_mat_eye(int rows, int cols, int type);
void* opencv_mat_diag(void* mat, int d);

/* Mat dump */
const char* opencv_mat_dump(void* mat);

/* ============================================================
 * Core arithmetic
 * ============================================================ */
int opencv_core_add(void* src1, void* src2, void* dst, void* mask, int dtype);
int opencv_core_subtract(void* src1, void* src2, void* dst, void* mask, int dtype);
int opencv_core_multiply(void* src1, void* src2, void* dst, double scale, int dtype);
int opencv_core_divide(void* src1, void* src2, void* dst, double scale, int dtype);
int opencv_core_add_weighted(void* src1, double alpha, void* src2, double beta, double gamma, void* dst, int dtype);
int opencv_core_scale_add(void* src1, double alpha, void* src2, void* dst);
int opencv_core_abs_diff(void* src1, void* src2, void* dst);

/* ============================================================
 * Core bitwise
 * ============================================================ */
int opencv_core_bitwise_and(void* src1, void* src2, void* dst, void* mask);
int opencv_core_bitwise_or(void* src1, void* src2, void* dst, void* mask);
int opencv_core_bitwise_xor(void* src1, void* src2, void* dst, void* mask);
int opencv_core_bitwise_not(void* src, void* dst, void* mask);

/* ============================================================
 * Core comparison
 * ============================================================ */
int opencv_core_compare(void* src1, void* src2, void* dst, int cmpop);
int opencv_core_min(void* src1, void* src2, void* dst);
int opencv_core_max(void* src1, void* src2, void* dst);
int opencv_core_in_range(void* src, void* lowerb, void* upperb, void* dst);

/* ============================================================
 * Core statistics
 * ============================================================ */
double opencv_core_norm(void* src, int normType, void* mask);
double opencv_core_norm_diff(void* src1, void* src2, int normType, void* mask);
int    opencv_core_mean(void* src, void* mask, double* result);
int    opencv_core_mean_std_dev(void* src, double* meanVal, double* stddev, void* mask);
int    opencv_core_min_max_loc(void* src, double* minVal, double* maxVal, int* minLoc, int* maxLoc, void* mask);
int    opencv_core_count_non_zero(void* src);
int    opencv_core_sum(void* src, double* result);

/* ============================================================
 * Core array operations
 * ============================================================ */
int opencv_core_normalize(void* src, void* dst, double alpha, double beta, int normType, int dtype, void* mask);
int opencv_core_flip(void* src, void* dst, int flipCode);
int opencv_core_rotate(void* src, void* dst, int rotateCode);
int opencv_core_transpose(void* src, void* dst);
int opencv_core_merge(void** mv, int count, void* dst);
int opencv_core_split(void* src, void*** mv, int* count);
void opencv_core_free_mat_array(void** mv, int count);
int opencv_core_convert_scale_abs(void* src, void* dst, double alpha, double beta);
int opencv_core_lut(void* src, void* lut, void* dst);
int opencv_core_copy_make_border(void* src, void* dst, int top, int bottom, int left, int right, int borderType, double s0, double s1, double s2, double s3);
int opencv_core_hconcat(void* src1, void* src2, void* dst);
int opencv_core_vconcat(void* src1, void* src2, void* dst);
int opencv_core_gemm(void* src1, void* src2, double alpha, void* src3, double beta, void* dst, int flags);
int opencv_core_dft(void* src, void* dst, int flags, int nonzeroRows);
int opencv_core_idft(void* src, void* dst, int flags, int nonzeroRows);

/* ============================================================
 * Imgproc - Color conversion
 * ============================================================ */
int opencv_imgproc_cvt_color(void* src, void* dst, int code, int dstCn);

/* ============================================================
 * Imgproc - Resize / Geometric
 * ============================================================ */
int opencv_imgproc_resize(void* src, void* dst, double dsize_w, double dsize_h, double fx, double fy, int interpolation);
int opencv_imgproc_warp_affine(void* src, void* dst, void* M, double dsize_w, double dsize_h, int flags, int borderMode, double bv0, double bv1, double bv2, double bv3);
int opencv_imgproc_warp_perspective(void* src, void* dst, void* M, double dsize_w, double dsize_h, int flags, int borderMode, double bv0, double bv1, double bv2, double bv3);
void* opencv_imgproc_get_rotation_matrix_2d(double center_x, double center_y, double angle, double scale);

/* ============================================================
 * Imgproc - Threshold
 * ============================================================ */
double opencv_imgproc_threshold(void* src, void* dst, double thresh, double maxval, int type);
int opencv_imgproc_adaptive_threshold(void* src, void* dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C);

/* ============================================================
 * Imgproc - Filtering
 * ============================================================ */
int opencv_imgproc_blur(void* src, void* dst, double ksize_w, double ksize_h, double anchor_x, double anchor_y, int borderType);
int opencv_imgproc_gaussian_blur(void* src, void* dst, double ksize_w, double ksize_h, double sigmaX, double sigmaY, int borderType);
int opencv_imgproc_median_blur(void* src, void* dst, int ksize);
int opencv_imgproc_bilateral_filter(void* src, void* dst, int d, double sigmaColor, double sigmaSpace, int borderType);
int opencv_imgproc_sobel(void* src, void* dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType);
int opencv_imgproc_scharr(void* src, void* dst, int ddepth, int dx, int dy, double scale, double delta, int borderType);
int opencv_imgproc_laplacian(void* src, void* dst, int ddepth, int ksize, double scale, double delta, int borderType);
int opencv_imgproc_canny(void* src, void* dst, double threshold1, double threshold2, int apertureSize, int L2gradient);

/* ============================================================
 * Imgproc - Morphology
 * ============================================================ */
int   opencv_imgproc_erode(void* src, void* dst, void* kernel, double anchor_x, double anchor_y, int iterations, int borderType, double bv0, double bv1, double bv2, double bv3);
int   opencv_imgproc_dilate(void* src, void* dst, void* kernel, double anchor_x, double anchor_y, int iterations, int borderType, double bv0, double bv1, double bv2, double bv3);
int   opencv_imgproc_morphology_ex(void* src, void* dst, int op, void* kernel, double anchor_x, double anchor_y, int iterations, int borderType, double bv0, double bv1, double bv2, double bv3);
void* opencv_imgproc_get_structuring_element(int shape, double ksize_w, double ksize_h, double anchor_x, double anchor_y);

/* ============================================================
 * Imgproc - Drawing
 * ============================================================ */
int opencv_imgproc_line(void* img, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double c0, double c1, double c2, double c3, int thickness, int lineType, int shift);
int opencv_imgproc_rectangle(void* img, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double c0, double c1, double c2, double c3, int thickness, int lineType, int shift);
int opencv_imgproc_circle(void* img, double center_x, double center_y, int radius, double c0, double c1, double c2, double c3, int thickness, int lineType, int shift);
int opencv_imgproc_ellipse(void* img, double center_x, double center_y, double axes_w, double axes_h, double angle, double startAngle, double endAngle, double c0, double c1, double c2, double c3, int thickness, int lineType, int shift);
int opencv_imgproc_put_text(void* img, const char* text, double org_x, double org_y, int fontFace, double fontScale, double c0, double c1, double c2, double c3, int thickness, int lineType, int bottomLeftOrigin);

/* ============================================================
 * Imgproc - Contours
 * ============================================================ */
int opencv_imgproc_find_contours(void* image, void* contours_mat, void* hierarchy, int mode, int method, double offset_x, double offset_y);
int opencv_imgproc_draw_contours(void* image, void* contours_mat, int contourIdx, double c0, double c1, double c2, double c3, int thickness, int lineType, void* hierarchy, int maxLevel, double offset_x, double offset_y);
double opencv_imgproc_contour_area(void* contour, int oriented);
double opencv_imgproc_arc_length(void* curve, int closed);
int opencv_imgproc_bounding_rect(void* points, int* x, int* y, int* width, int* height);

/* ============================================================
 * Imgproc - Histogram
 * ============================================================ */
int opencv_imgproc_equalize_hist(void* src, void* dst);
int opencv_imgproc_calc_hist_1(void* image, void* hist,
    void* channels_mat, void* mask, void* hist_size_mat, void* ranges_mat,
    int uniform, int accumulate);
double opencv_imgproc_compare_hist(void* h1, void* h2, int method);

/* ============================================================
 * Imgproc - Additional feature detection (used by video4j)
 * ============================================================ */
int opencv_imgproc_corner_harris(void* src, void* dst, int blockSize, int ksize, double k, int borderType);
int opencv_imgproc_good_features_to_track(void* image, void* corners,
    int maxCorners, double qualityLevel, double minDistance,
    void* mask, int blockSize, int gradientSize, int useHarrisDetector, double k);
int opencv_imgproc_hough_lines(void* image, void* lines, double rho, double theta, int threshold, double srn, double stn);
int opencv_imgproc_hough_lines_p(void* image, void* lines, double rho, double theta, int threshold, double minLineLength, double maxLineGap);
int opencv_imgproc_get_rect_sub_pix(void* image, double patchSize_w, double patchSize_h, double center_x, double center_y, void* patch, int patchType);

/* ============================================================
 * Video - Optical flow
 * ============================================================ */
int opencv_video_calc_optical_flow_pyr_lk(
    void* prev_img,
    void* next_img,
    void* prev_pts,
    void* next_pts,
    void* status,
    void* err,
    double win_size_w,
    double win_size_h,
    int max_level,
    int criteria_type,
    int criteria_max_count,
    double criteria_epsilon,
    int flags,
    double min_eig_threshold);

/* ============================================================
 * Core - Additional math (used by video4j)
 * ============================================================ */
int opencv_core_sqrt(void* src, void* dst);

/* ============================================================
 * VideoCapture (videoio module)
 * ============================================================ */
void*  opencv_videocapture_create(void);
void   opencv_videocapture_delete(void* cap);
int    opencv_videocapture_open_file(void* cap, const char* filename, int apiPreference);
int    opencv_videocapture_open_device(void* cap, int index, int apiPreference);
int    opencv_videocapture_is_opened(void* cap);
void   opencv_videocapture_release(void* cap);
int    opencv_videocapture_read(void* cap, void* image);
int    opencv_videocapture_grab(void* cap);
int    opencv_videocapture_retrieve(void* cap, void* image, int flag);
int    opencv_videocapture_set(void* cap, int propId, double value);
double opencv_videocapture_get(void* cap, int propId);

/* ============================================================
 * CascadeClassifier (objdetect module)
 * ============================================================ */
void* opencv_cascade_create(void);
void  opencv_cascade_delete(void* cascade);
int   opencv_cascade_load(void* cascade, const char* path);

/* Detect faces. out_rects is a flat int array [x0,y0,w0,h0, x1,...].
 * Returns number of detections (up to max_count), or -1 on error. */
int   opencv_cascade_detect(void* cascade, void* mat, int* out_rects, int max_count);

/* Detect faces with parameters. flags=4 is CASCADE_SCALE_IMAGE. */
int   opencv_cascade_detect_params(void* cascade, void* mat,
          double scale_factor, int min_neighbors, int flags,
          int min_size_w, int min_size_h,
          int* out_rects, int max_count);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_FFM_H */
