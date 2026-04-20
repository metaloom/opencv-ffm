#include "opencv_ffm.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <cstring>

/* ============================================================
 * Thread-local error handling
 * ============================================================ */
static thread_local std::string g_last_error;

static void set_last_error(const char* msg) {
    g_last_error = msg ? msg : "unknown error";
}

static void clear_error() {
    g_last_error.clear();
}

#define MAT(p) (reinterpret_cast<cv::Mat*>(p))
#define CMAT(p) (reinterpret_cast<const cv::Mat*>(p))
#define CAP(p) (reinterpret_cast<cv::VideoCapture*>(p))

#define TRY try { clear_error();
#define CATCH_RET(ret) } catch (const cv::Exception& e) { set_last_error(e.what()); return ret; } \
    catch (const std::exception& e) { set_last_error(e.what()); return ret; }
#define CATCH_RET_INT CATCH_RET(-1)
#define CATCH_RET_NULL CATCH_RET(nullptr)
#define CATCH_RET_ZERO CATCH_RET(0)

extern "C" {

/* ============================================================
 * Error handling
 * ============================================================ */
const char* opencv_get_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

/* ============================================================
 * Mat lifecycle
 * ============================================================ */
void* opencv_mat_create(void) {
    TRY
    return new cv::Mat();
    CATCH_RET_NULL
}

void* opencv_mat_create_with_size(int rows, int cols, int type) {
    TRY
    return new cv::Mat(rows, cols, type);
    CATCH_RET_NULL
}

void* opencv_mat_create_with_scalar(int rows, int cols, int type, double s0, double s1, double s2, double s3) {
    TRY
    return new cv::Mat(rows, cols, type, cv::Scalar(s0, s1, s2, s3));
    CATCH_RET_NULL
}

void* opencv_mat_create_from_data(int rows, int cols, int type, void* data, long step) {
    TRY
    // Creates a header that points to existing data (no copy).
    // The caller must ensure data outlives this Mat.
    cv::Mat header(rows, cols, type, data, (size_t)step);
    return new cv::Mat(header.clone()); // clone to own the data
    CATCH_RET_NULL
}

void opencv_mat_delete(void* mat) {
    delete MAT(mat);
}

void opencv_mat_release(void* mat) {
    if (mat) MAT(mat)->release();
}

void* opencv_mat_clone(void* mat) {
    TRY
    return new cv::Mat(MAT(mat)->clone());
    CATCH_RET_NULL
}

/* ============================================================
 * Mat metadata
 * ============================================================ */
int opencv_mat_rows(void* mat) {
    return mat ? MAT(mat)->rows : 0;
}

int opencv_mat_cols(void* mat) {
    return mat ? MAT(mat)->cols : 0;
}

int opencv_mat_type(void* mat) {
    return mat ? MAT(mat)->type() : 0;
}

int opencv_mat_depth(void* mat) {
    return mat ? MAT(mat)->depth() : 0;
}

int opencv_mat_channels(void* mat) {
    return mat ? MAT(mat)->channels() : 0;
}

int opencv_mat_dims(void* mat) {
    return mat ? MAT(mat)->dims : 0;
}

int opencv_mat_empty(void* mat) {
    return mat ? (MAT(mat)->empty() ? 1 : 0) : 1;
}

long opencv_mat_total(void* mat) {
    return mat ? (long)MAT(mat)->total() : 0;
}

long opencv_mat_elem_size(void* mat) {
    return mat ? (long)MAT(mat)->elemSize() : 0;
}

long opencv_mat_elem_size1(void* mat) {
    return mat ? (long)MAT(mat)->elemSize1() : 0;
}

long opencv_mat_step1(void* mat, int i) {
    return mat ? (long)MAT(mat)->step1(i) : 0;
}

void* opencv_mat_data_addr(void* mat) {
    return mat ? (void*)MAT(mat)->data : nullptr;
}

int opencv_mat_is_continuous(void* mat) {
    return mat ? (MAT(mat)->isContinuous() ? 1 : 0) : 0;
}

int opencv_mat_is_submatrix(void* mat) {
    return mat ? (MAT(mat)->isSubmatrix() ? 1 : 0) : 0;
}

/* ============================================================
 * Mat data access
 * ============================================================ */
static int mat_put(cv::Mat* m, int row, int col, int count, const void* data, int elemSize) {
    if (!m || !data) return -1;
    if (row < 0 || row >= m->rows || col < 0 || col >= m->cols) return -1;

    int rest = ((m->rows - row) * m->cols - col) * m->channels();
    if (count > rest) count = rest;

    int cn = m->channels();
    int totalBytes = count * elemSize;
    if (m->isContinuous()) {
        memcpy(m->data + (row * m->cols + col) * m->elemSize(), data, totalBytes);
    } else {
        // Row-by-row copy
        int colOffset = col * cn;
        int copied = 0;
        for (int r = row; r < m->rows && copied < count; r++) {
            uchar* rowPtr = m->ptr(r);
            int startCh = (r == row) ? colOffset : 0;
            int endCh = m->cols * cn;
            int toCopy = std::min(count - copied, endCh - startCh);
            memcpy(rowPtr + startCh * elemSize, (const char*)data + copied * elemSize, toCopy * elemSize);
            copied += toCopy;
        }
    }
    return count;
}

static int mat_get(const cv::Mat* m, int row, int col, int count, void* data, int elemSize) {
    if (!m || !data) return -1;
    if (row < 0 || row >= m->rows || col < 0 || col >= m->cols) return -1;

    int rest = ((m->rows - row) * m->cols - col) * m->channels();
    if (count > rest) count = rest;

    int cn = m->channels();
    if (m->isContinuous()) {
        memcpy(data, m->data + (row * m->cols + col) * m->elemSize(), count * elemSize);
    } else {
        int colOffset = col * cn;
        int copied = 0;
        for (int r = row; r < m->rows && copied < count; r++) {
            const uchar* rowPtr = m->ptr(r);
            int startCh = (r == row) ? colOffset : 0;
            int endCh = m->cols * cn;
            int toCopy = std::min(count - copied, endCh - startCh);
            memcpy((char*)data + copied * elemSize, rowPtr + startCh * elemSize, toCopy * elemSize);
            copied += toCopy;
        }
    }
    return count;
}

int opencv_mat_put_b(void* mat, int row, int col, int count, const char* data) {
    TRY return mat_put(MAT(mat), row, col, count, data, 1); CATCH_RET_INT
}

int opencv_mat_put_s(void* mat, int row, int col, int count, const short* data) {
    TRY return mat_put(MAT(mat), row, col, count, data, 2); CATCH_RET_INT
}

int opencv_mat_put_i(void* mat, int row, int col, int count, const int* data) {
    TRY return mat_put(MAT(mat), row, col, count, data, 4); CATCH_RET_INT
}

int opencv_mat_put_f(void* mat, int row, int col, int count, const float* data) {
    TRY return mat_put(MAT(mat), row, col, count, data, 4); CATCH_RET_INT
}

int opencv_mat_put_d(void* mat, int row, int col, int count, const double* data) {
    TRY return mat_put(MAT(mat), row, col, count, data, 8); CATCH_RET_INT
}

int opencv_mat_get_b(void* mat, int row, int col, int count, char* data) {
    TRY return mat_get(CMAT(mat), row, col, count, data, 1); CATCH_RET_INT
}

int opencv_mat_get_s(void* mat, int row, int col, int count, short* data) {
    TRY return mat_get(CMAT(mat), row, col, count, data, 2); CATCH_RET_INT
}

int opencv_mat_get_i(void* mat, int row, int col, int count, int* data) {
    TRY return mat_get(CMAT(mat), row, col, count, data, 4); CATCH_RET_INT
}

int opencv_mat_get_f(void* mat, int row, int col, int count, float* data) {
    TRY return mat_get(CMAT(mat), row, col, count, data, 4); CATCH_RET_INT
}

int opencv_mat_get_d(void* mat, int row, int col, int count, double* data) {
    TRY return mat_get(CMAT(mat), row, col, count, data, 8); CATCH_RET_INT
}

int opencv_mat_get_pixel(void* mat, int row, int col, double* values) {
    TRY
    cv::Mat* m = MAT(mat);
    if (!m || row < 0 || row >= m->rows || col < 0 || col >= m->cols) return -1;
    int cn = m->channels();
    for (int c = 0; c < cn && c < 4; c++) {
        switch (m->depth()) {
            case CV_8U:  values[c] = m->ptr<uchar>(row)[col * cn + c]; break;
            case CV_8S:  values[c] = m->ptr<schar>(row)[col * cn + c]; break;
            case CV_16U: values[c] = m->ptr<ushort>(row)[col * cn + c]; break;
            case CV_16S: values[c] = m->ptr<short>(row)[col * cn + c]; break;
            case CV_32S: values[c] = m->ptr<int>(row)[col * cn + c]; break;
            case CV_32F: values[c] = m->ptr<float>(row)[col * cn + c]; break;
            case CV_64F: values[c] = m->ptr<double>(row)[col * cn + c]; break;
            default: values[c] = 0;
        }
    }
    return cn;
    CATCH_RET_INT
}

/* ============================================================
 * Mat operations
 * ============================================================ */
int opencv_mat_copy_to(void* src, void* dst) {
    TRY MAT(src)->copyTo(*MAT(dst)); return 0; CATCH_RET_INT
}

int opencv_mat_copy_to_masked(void* src, void* dst, void* mask) {
    TRY MAT(src)->copyTo(*MAT(dst), *MAT(mask)); return 0; CATCH_RET_INT
}

int opencv_mat_convert_to(void* src, void* dst, int rtype, double alpha, double beta) {
    TRY MAT(src)->convertTo(*MAT(dst), rtype, alpha, beta); return 0; CATCH_RET_INT
}

void* opencv_mat_set_to_scalar(void* mat, double s0, double s1, double s2, double s3) {
    TRY MAT(mat)->setTo(cv::Scalar(s0, s1, s2, s3)); return mat; CATCH_RET_NULL
}

void* opencv_mat_set_to_masked(void* mat, double s0, double s1, double s2, double s3, void* mask) {
    TRY MAT(mat)->setTo(cv::Scalar(s0, s1, s2, s3), *MAT(mask)); return mat; CATCH_RET_NULL
}

void* opencv_mat_reshape(void* mat, int cn, int rows) {
    TRY return new cv::Mat(MAT(mat)->reshape(cn, rows)); CATCH_RET_NULL
}

void* opencv_mat_row(void* mat, int y) {
    TRY return new cv::Mat(MAT(mat)->row(y)); CATCH_RET_NULL
}

void* opencv_mat_col(void* mat, int x) {
    TRY return new cv::Mat(MAT(mat)->col(x)); CATCH_RET_NULL
}

void* opencv_mat_row_range(void* mat, int startrow, int endrow) {
    TRY return new cv::Mat(MAT(mat)->rowRange(startrow, endrow)); CATCH_RET_NULL
}

void* opencv_mat_col_range(void* mat, int startcol, int endcol) {
    TRY return new cv::Mat(MAT(mat)->colRange(startcol, endcol)); CATCH_RET_NULL
}

void* opencv_mat_submat(void* mat, int rowStart, int rowEnd, int colStart, int colEnd) {
    TRY return new cv::Mat((*MAT(mat))(cv::Range(rowStart, rowEnd), cv::Range(colStart, colEnd))); CATCH_RET_NULL
}

void* opencv_mat_t(void* mat) {
    TRY return new cv::Mat(MAT(mat)->t()); CATCH_RET_NULL
}

void* opencv_mat_inv(void* mat, int method) {
    TRY return new cv::Mat(MAT(mat)->inv(method)); CATCH_RET_NULL
}

void* opencv_mat_mul(void* mat, void* other, double scale) {
    TRY return new cv::Mat(MAT(mat)->mul(*MAT(other), scale)); CATCH_RET_NULL
}

void* opencv_mat_mat_mul(void* mat, void* other) {
    TRY return new cv::Mat((*MAT(mat)) * (*MAT(other))); CATCH_RET_NULL
}

double opencv_mat_dot(void* mat, void* other) {
    TRY return MAT(mat)->dot(*MAT(other)); CATCH_RET(0.0)
}

void* opencv_mat_cross(void* mat, void* other) {
    TRY return new cv::Mat(MAT(mat)->cross(*MAT(other))); CATCH_RET_NULL
}

int opencv_mat_push_back(void* mat, void* other) {
    TRY MAT(mat)->push_back(*MAT(other)); return 0; CATCH_RET_INT
}

void opencv_mat_create_inplace(void* mat, int rows, int cols, int type) {
    if (mat) MAT(mat)->create(rows, cols, type);
}

int opencv_mat_check_vector(void* mat, int elemChannels, int depth, int requireContinuous) {
    if (!mat) return -1;
    return MAT(mat)->checkVector(elemChannels, depth, requireContinuous != 0);
}

/* Mat factories */
void* opencv_mat_zeros(int rows, int cols, int type) {
    TRY return new cv::Mat(cv::Mat::zeros(rows, cols, type)); CATCH_RET_NULL
}

void* opencv_mat_ones(int rows, int cols, int type) {
    TRY return new cv::Mat(cv::Mat::ones(rows, cols, type)); CATCH_RET_NULL
}

void* opencv_mat_eye(int rows, int cols, int type) {
    TRY return new cv::Mat(cv::Mat::eye(rows, cols, type)); CATCH_RET_NULL
}

void* opencv_mat_diag(void* mat, int d) {
    TRY return new cv::Mat(MAT(mat)->diag(d)); CATCH_RET_NULL
}

/* Mat dump - returns a thread-local string */
static thread_local std::string g_dump_result;

const char* opencv_mat_dump(void* mat) {
    TRY
    std::ostringstream oss;
    oss << *MAT(mat);
    g_dump_result = oss.str();
    return g_dump_result.c_str();
    CATCH_RET(nullptr)
}

/* ============================================================
 * Core arithmetic
 * ============================================================ */
int opencv_core_add(void* src1, void* src2, void* dst, void* mask, int dtype) {
    TRY
    cv::add(*MAT(src1), *MAT(src2), *MAT(dst), mask ? *MAT(mask) : cv::noArray(), dtype);
    return 0;
    CATCH_RET_INT
}

int opencv_core_subtract(void* src1, void* src2, void* dst, void* mask, int dtype) {
    TRY
    cv::subtract(*MAT(src1), *MAT(src2), *MAT(dst), mask ? *MAT(mask) : cv::noArray(), dtype);
    return 0;
    CATCH_RET_INT
}

int opencv_core_multiply(void* src1, void* src2, void* dst, double scale, int dtype) {
    TRY
    cv::multiply(*MAT(src1), *MAT(src2), *MAT(dst), scale, dtype);
    return 0;
    CATCH_RET_INT
}

int opencv_core_divide(void* src1, void* src2, void* dst, double scale, int dtype) {
    TRY
    cv::divide(*MAT(src1), *MAT(src2), *MAT(dst), scale, dtype);
    return 0;
    CATCH_RET_INT
}

int opencv_core_add_weighted(void* src1, double alpha, void* src2, double beta, double gamma, void* dst, int dtype) {
    TRY
    cv::addWeighted(*MAT(src1), alpha, *MAT(src2), beta, gamma, *MAT(dst), dtype);
    return 0;
    CATCH_RET_INT
}

int opencv_core_scale_add(void* src1, double alpha, void* src2, void* dst) {
    TRY
    cv::scaleAdd(*MAT(src1), alpha, *MAT(src2), *MAT(dst));
    return 0;
    CATCH_RET_INT
}

int opencv_core_abs_diff(void* src1, void* src2, void* dst) {
    TRY
    cv::absdiff(*MAT(src1), *MAT(src2), *MAT(dst));
    return 0;
    CATCH_RET_INT
}

/* ============================================================
 * Core bitwise
 * ============================================================ */
int opencv_core_bitwise_and(void* src1, void* src2, void* dst, void* mask) {
    TRY
    cv::bitwise_and(*MAT(src1), *MAT(src2), *MAT(dst), mask ? *MAT(mask) : cv::noArray());
    return 0;
    CATCH_RET_INT
}

int opencv_core_bitwise_or(void* src1, void* src2, void* dst, void* mask) {
    TRY
    cv::bitwise_or(*MAT(src1), *MAT(src2), *MAT(dst), mask ? *MAT(mask) : cv::noArray());
    return 0;
    CATCH_RET_INT
}

int opencv_core_bitwise_xor(void* src1, void* src2, void* dst, void* mask) {
    TRY
    cv::bitwise_xor(*MAT(src1), *MAT(src2), *MAT(dst), mask ? *MAT(mask) : cv::noArray());
    return 0;
    CATCH_RET_INT
}

int opencv_core_bitwise_not(void* src, void* dst, void* mask) {
    TRY
    cv::bitwise_not(*MAT(src), *MAT(dst), mask ? *MAT(mask) : cv::noArray());
    return 0;
    CATCH_RET_INT
}

/* ============================================================
 * Core comparison
 * ============================================================ */
int opencv_core_compare(void* src1, void* src2, void* dst, int cmpop) {
    TRY cv::compare(*MAT(src1), *MAT(src2), *MAT(dst), cmpop); return 0; CATCH_RET_INT
}

int opencv_core_min(void* src1, void* src2, void* dst) {
    TRY cv::min(*MAT(src1), *MAT(src2), *MAT(dst)); return 0; CATCH_RET_INT
}

int opencv_core_max(void* src1, void* src2, void* dst) {
    TRY cv::max(*MAT(src1), *MAT(src2), *MAT(dst)); return 0; CATCH_RET_INT
}

int opencv_core_in_range(void* src, void* lowerb, void* upperb, void* dst) {
    TRY cv::inRange(*MAT(src), *MAT(lowerb), *MAT(upperb), *MAT(dst)); return 0; CATCH_RET_INT
}

/* ============================================================
 * Core statistics
 * ============================================================ */
double opencv_core_norm(void* src, int normType, void* mask) {
    TRY return cv::norm(*MAT(src), normType, mask ? *MAT(mask) : cv::noArray()); CATCH_RET(0.0)
}

double opencv_core_norm_diff(void* src1, void* src2, int normType, void* mask) {
    TRY return cv::norm(*MAT(src1), *MAT(src2), normType, mask ? *MAT(mask) : cv::noArray()); CATCH_RET(0.0)
}

int opencv_core_mean(void* src, void* mask, double* result) {
    TRY
    cv::Scalar s = cv::mean(*MAT(src), mask ? *MAT(mask) : cv::noArray());
    result[0] = s[0]; result[1] = s[1]; result[2] = s[2]; result[3] = s[3];
    return 0;
    CATCH_RET_INT
}

int opencv_core_mean_std_dev(void* src, double* meanVal, double* stddev, void* mask) {
    TRY
    cv::Scalar meanScalar, stdScalar;
    cv::meanStdDev(*MAT(src), meanScalar, stdScalar, mask ? *MAT(mask) : cv::noArray());
    for (int i = 0; i < 4; i++) {
        meanVal[i] = meanScalar[i];
        stddev[i] = stdScalar[i];
    }
    return 0;
    CATCH_RET_INT
}

int opencv_core_min_max_loc(void* src, double* minVal, double* maxVal, int* minLoc, int* maxLoc, void* mask) {
    TRY
    cv::Point minP, maxP;
    cv::minMaxLoc(*MAT(src), minVal, maxVal, &minP, &maxP, mask ? *MAT(mask) : cv::noArray());
    if (minLoc) { minLoc[0] = minP.x; minLoc[1] = minP.y; }
    if (maxLoc) { maxLoc[0] = maxP.x; maxLoc[1] = maxP.y; }
    return 0;
    CATCH_RET_INT
}

int opencv_core_count_non_zero(void* src) {
    TRY return cv::countNonZero(*MAT(src)); CATCH_RET_INT
}

int opencv_core_sum(void* src, double* result) {
    TRY
    cv::Scalar s = cv::sum(*MAT(src));
    result[0] = s[0]; result[1] = s[1]; result[2] = s[2]; result[3] = s[3];
    return 0;
    CATCH_RET_INT
}

/* ============================================================
 * Core array operations
 * ============================================================ */
int opencv_core_normalize(void* src, void* dst, double alpha, double beta, int normType, int dtype, void* mask) {
    TRY
    cv::normalize(*MAT(src), *MAT(dst), alpha, beta, normType, dtype, mask ? *MAT(mask) : cv::noArray());
    return 0;
    CATCH_RET_INT
}

int opencv_core_flip(void* src, void* dst, int flipCode) {
    TRY cv::flip(*MAT(src), *MAT(dst), flipCode); return 0; CATCH_RET_INT
}

int opencv_core_rotate(void* src, void* dst, int rotateCode) {
    TRY cv::rotate(*MAT(src), *MAT(dst), rotateCode); return 0; CATCH_RET_INT
}

int opencv_core_transpose(void* src, void* dst) {
    TRY cv::transpose(*MAT(src), *MAT(dst)); return 0; CATCH_RET_INT
}

int opencv_core_merge(void** mv, int count, void* dst) {
    TRY
    std::vector<cv::Mat> mats(count);
    for (int i = 0; i < count; i++) mats[i] = *MAT(mv[i]);
    cv::merge(mats, *MAT(dst));
    return 0;
    CATCH_RET_INT
}

int opencv_core_split(void* src, void*** mv, int* count) {
    TRY
    std::vector<cv::Mat> planes;
    cv::split(*MAT(src), planes);
    *count = (int)planes.size();
    *mv = new void*[planes.size()];
    for (size_t i = 0; i < planes.size(); i++) {
        (*mv)[i] = new cv::Mat(planes[i]);
    }
    return 0;
    CATCH_RET_INT
}

void opencv_core_free_mat_array(void** mv, int count) {
    if (!mv) return;
    for (int i = 0; i < count; i++) {
        delete MAT(mv[i]);
    }
    delete[] mv;
}

int opencv_core_convert_scale_abs(void* src, void* dst, double alpha, double beta) {
    TRY cv::convertScaleAbs(*MAT(src), *MAT(dst), alpha, beta); return 0; CATCH_RET_INT
}

int opencv_core_lut(void* src, void* lut, void* dst) {
    TRY cv::LUT(*MAT(src), *MAT(lut), *MAT(dst)); return 0; CATCH_RET_INT
}

int opencv_core_copy_make_border(void* src, void* dst, int top, int bottom, int left, int right, int borderType, double s0, double s1, double s2, double s3) {
    TRY cv::copyMakeBorder(*MAT(src), *MAT(dst), top, bottom, left, right, borderType, cv::Scalar(s0, s1, s2, s3)); return 0; CATCH_RET_INT
}

int opencv_core_hconcat(void* src1, void* src2, void* dst) {
    TRY cv::hconcat(*MAT(src1), *MAT(src2), *MAT(dst)); return 0; CATCH_RET_INT
}

int opencv_core_vconcat(void* src1, void* src2, void* dst) {
    TRY cv::vconcat(*MAT(src1), *MAT(src2), *MAT(dst)); return 0; CATCH_RET_INT
}

int opencv_core_gemm(void* src1, void* src2, double alpha, void* src3, double beta, void* dst, int flags) {
    TRY cv::gemm(*MAT(src1), *MAT(src2), alpha, *MAT(src3), beta, *MAT(dst), flags); return 0; CATCH_RET_INT
}

int opencv_core_dft(void* src, void* dst, int flags, int nonzeroRows) {
    TRY cv::dft(*MAT(src), *MAT(dst), flags, nonzeroRows); return 0; CATCH_RET_INT
}

int opencv_core_idft(void* src, void* dst, int flags, int nonzeroRows) {
    TRY cv::idft(*MAT(src), *MAT(dst), flags, nonzeroRows); return 0; CATCH_RET_INT
}

/* ============================================================
 * Imgproc - Color conversion
 * ============================================================ */
int opencv_imgproc_cvt_color(void* src, void* dst, int code, int dstCn) {
    TRY cv::cvtColor(*MAT(src), *MAT(dst), code, dstCn); return 0; CATCH_RET_INT
}

/* ============================================================
 * Imgproc - Resize / Geometric
 * ============================================================ */
int opencv_imgproc_resize(void* src, void* dst, double dsize_w, double dsize_h, double fx, double fy, int interpolation) {
    TRY cv::resize(*MAT(src), *MAT(dst), cv::Size((int)dsize_w, (int)dsize_h), fx, fy, interpolation); return 0; CATCH_RET_INT
}

int opencv_imgproc_warp_affine(void* src, void* dst, void* M, double dsize_w, double dsize_h, int flags, int borderMode, double bv0, double bv1, double bv2, double bv3) {
    TRY cv::warpAffine(*MAT(src), *MAT(dst), *MAT(M), cv::Size((int)dsize_w, (int)dsize_h), flags, borderMode, cv::Scalar(bv0, bv1, bv2, bv3)); return 0; CATCH_RET_INT
}

int opencv_imgproc_warp_perspective(void* src, void* dst, void* M, double dsize_w, double dsize_h, int flags, int borderMode, double bv0, double bv1, double bv2, double bv3) {
    TRY cv::warpPerspective(*MAT(src), *MAT(dst), *MAT(M), cv::Size((int)dsize_w, (int)dsize_h), flags, borderMode, cv::Scalar(bv0, bv1, bv2, bv3)); return 0; CATCH_RET_INT
}

void* opencv_imgproc_get_rotation_matrix_2d(double center_x, double center_y, double angle, double scale) {
    TRY return new cv::Mat(cv::getRotationMatrix2D(cv::Point2f((float)center_x, (float)center_y), angle, scale)); CATCH_RET_NULL
}

/* ============================================================
 * Imgproc - Threshold
 * ============================================================ */
double opencv_imgproc_threshold(void* src, void* dst, double thresh, double maxval, int type) {
    TRY return cv::threshold(*MAT(src), *MAT(dst), thresh, maxval, type); CATCH_RET(0.0)
}

int opencv_imgproc_adaptive_threshold(void* src, void* dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C) {
    TRY cv::adaptiveThreshold(*MAT(src), *MAT(dst), maxValue, adaptiveMethod, thresholdType, blockSize, C); return 0; CATCH_RET_INT
}

/* ============================================================
 * Imgproc - Filtering
 * ============================================================ */
int opencv_imgproc_blur(void* src, void* dst, double ksize_w, double ksize_h, double anchor_x, double anchor_y, int borderType) {
    TRY cv::blur(*MAT(src), *MAT(dst), cv::Size((int)ksize_w, (int)ksize_h), cv::Point((int)anchor_x, (int)anchor_y), borderType); return 0; CATCH_RET_INT
}

int opencv_imgproc_gaussian_blur(void* src, void* dst, double ksize_w, double ksize_h, double sigmaX, double sigmaY, int borderType) {
    TRY cv::GaussianBlur(*MAT(src), *MAT(dst), cv::Size((int)ksize_w, (int)ksize_h), sigmaX, sigmaY, borderType); return 0; CATCH_RET_INT
}

int opencv_imgproc_median_blur(void* src, void* dst, int ksize) {
    TRY cv::medianBlur(*MAT(src), *MAT(dst), ksize); return 0; CATCH_RET_INT
}

int opencv_imgproc_bilateral_filter(void* src, void* dst, int d, double sigmaColor, double sigmaSpace, int borderType) {
    TRY cv::bilateralFilter(*MAT(src), *MAT(dst), d, sigmaColor, sigmaSpace, borderType); return 0; CATCH_RET_INT
}

int opencv_imgproc_sobel(void* src, void* dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType) {
    TRY cv::Sobel(*MAT(src), *MAT(dst), ddepth, dx, dy, ksize, scale, delta, borderType); return 0; CATCH_RET_INT
}

int opencv_imgproc_scharr(void* src, void* dst, int ddepth, int dx, int dy, double scale, double delta, int borderType) {
    TRY cv::Scharr(*MAT(src), *MAT(dst), ddepth, dx, dy, scale, delta, borderType); return 0; CATCH_RET_INT
}

int opencv_imgproc_laplacian(void* src, void* dst, int ddepth, int ksize, double scale, double delta, int borderType) {
    TRY cv::Laplacian(*MAT(src), *MAT(dst), ddepth, ksize, scale, delta, borderType); return 0; CATCH_RET_INT
}

int opencv_imgproc_canny(void* src, void* dst, double threshold1, double threshold2, int apertureSize, int L2gradient) {
    TRY cv::Canny(*MAT(src), *MAT(dst), threshold1, threshold2, apertureSize, L2gradient != 0); return 0; CATCH_RET_INT
}

/* ============================================================
 * Imgproc - Morphology
 * ============================================================ */
int opencv_imgproc_erode(void* src, void* dst, void* kernel, double anchor_x, double anchor_y, int iterations, int borderType, double bv0, double bv1, double bv2, double bv3) {
    TRY cv::erode(*MAT(src), *MAT(dst), *MAT(kernel), cv::Point((int)anchor_x, (int)anchor_y), iterations, borderType, cv::Scalar(bv0, bv1, bv2, bv3)); return 0; CATCH_RET_INT
}

int opencv_imgproc_dilate(void* src, void* dst, void* kernel, double anchor_x, double anchor_y, int iterations, int borderType, double bv0, double bv1, double bv2, double bv3) {
    TRY cv::dilate(*MAT(src), *MAT(dst), *MAT(kernel), cv::Point((int)anchor_x, (int)anchor_y), iterations, borderType, cv::Scalar(bv0, bv1, bv2, bv3)); return 0; CATCH_RET_INT
}

int opencv_imgproc_morphology_ex(void* src, void* dst, int op, void* kernel, double anchor_x, double anchor_y, int iterations, int borderType, double bv0, double bv1, double bv2, double bv3) {
    TRY cv::morphologyEx(*MAT(src), *MAT(dst), op, *MAT(kernel), cv::Point((int)anchor_x, (int)anchor_y), iterations, borderType, cv::Scalar(bv0, bv1, bv2, bv3)); return 0; CATCH_RET_INT
}

void* opencv_imgproc_get_structuring_element(int shape, double ksize_w, double ksize_h, double anchor_x, double anchor_y) {
    TRY return new cv::Mat(cv::getStructuringElement(shape, cv::Size((int)ksize_w, (int)ksize_h), cv::Point((int)anchor_x, (int)anchor_y))); CATCH_RET_NULL
}

/* ============================================================
 * Imgproc - Drawing
 * ============================================================ */
int opencv_imgproc_line(void* img, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double c0, double c1, double c2, double c3, int thickness, int lineType, int shift) {
    TRY cv::line(*MAT(img), cv::Point((int)pt1_x, (int)pt1_y), cv::Point((int)pt2_x, (int)pt2_y), cv::Scalar(c0, c1, c2, c3), thickness, lineType, shift); return 0; CATCH_RET_INT
}

int opencv_imgproc_rectangle(void* img, double pt1_x, double pt1_y, double pt2_x, double pt2_y, double c0, double c1, double c2, double c3, int thickness, int lineType, int shift) {
    TRY cv::rectangle(*MAT(img), cv::Point((int)pt1_x, (int)pt1_y), cv::Point((int)pt2_x, (int)pt2_y), cv::Scalar(c0, c1, c2, c3), thickness, lineType, shift); return 0; CATCH_RET_INT
}

int opencv_imgproc_circle(void* img, double center_x, double center_y, int radius, double c0, double c1, double c2, double c3, int thickness, int lineType, int shift) {
    TRY cv::circle(*MAT(img), cv::Point((int)center_x, (int)center_y), radius, cv::Scalar(c0, c1, c2, c3), thickness, lineType, shift); return 0; CATCH_RET_INT
}

int opencv_imgproc_ellipse(void* img, double center_x, double center_y, double axes_w, double axes_h, double angle, double startAngle, double endAngle, double c0, double c1, double c2, double c3, int thickness, int lineType, int shift) {
    TRY cv::ellipse(*MAT(img), cv::Point((int)center_x, (int)center_y), cv::Size((int)axes_w, (int)axes_h), angle, startAngle, endAngle, cv::Scalar(c0, c1, c2, c3), thickness, lineType, shift); return 0; CATCH_RET_INT
}

int opencv_imgproc_put_text(void* img, const char* text, double org_x, double org_y, int fontFace, double fontScale, double c0, double c1, double c2, double c3, int thickness, int lineType, int bottomLeftOrigin) {
    TRY cv::putText(*MAT(img), text, cv::Point((int)org_x, (int)org_y), fontFace, fontScale, cv::Scalar(c0, c1, c2, c3), thickness, lineType, bottomLeftOrigin != 0); return 0; CATCH_RET_INT
}

/* ============================================================
 * Imgproc - Contours
 * ============================================================ */
int opencv_imgproc_find_contours(void* image, void* contours_mat, void* hierarchy, int mode, int method, double offset_x, double offset_y) {
    TRY
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(*MAT(image), contours, *MAT(hierarchy), mode, method, cv::Point((int)offset_x, (int)offset_y));
    // Serialize contours into the output Mat as a flat structure
    // Each contour is stored as a row in contours_mat
    // For simplicity, store all points flat with contour sizes
    int totalPoints = 0;
    for (auto& c : contours) totalPoints += (int)c.size();
    if (totalPoints > 0) {
        // Store as Nx1 2-channel Mat (x,y pairs) with contour boundaries tracked via hierarchy
        cv::Mat& cmat = *MAT(contours_mat);
        cmat.create((int)contours.size(), 1, CV_32SC1);
        // Actually, store contour count and each contour as separate Mat
        // Use a simpler approach: store vector of Mats
        std::vector<cv::Mat> cmats(contours.size());
        for (size_t i = 0; i < contours.size(); i++) {
            cmats[i] = cv::Mat(contours[i]).reshape(1);
        }
        // Combine into single mat for transfer
        if (!cmats.empty()) {
            cv::vconcat(cmats, cmat);
        }
    }
    return (int)contours.size();
    CATCH_RET_INT
}

int opencv_imgproc_draw_contours(void* image, void* contours_mat, int contourIdx, double c0, double c1, double c2, double c3, int thickness, int lineType, void* hierarchy, int maxLevel, double offset_x, double offset_y) {
    TRY
    // For a simplified version, we treat contours_mat as containing points
    // A more complete implementation would deserialize the contours
    std::vector<std::vector<cv::Point>> contours;
    // Simple case: single contour from Mat
    cv::Mat& cmat = *MAT(contours_mat);
    if (!cmat.empty()) {
        std::vector<cv::Point> pts;
        for (int i = 0; i < cmat.rows; i++) {
            pts.push_back(cv::Point(cmat.at<int>(i, 0), cmat.at<int>(i, 1)));
        }
        contours.push_back(pts);
    }
    cv::drawContours(*MAT(image), contours, contourIdx, cv::Scalar(c0, c1, c2, c3), thickness, lineType,
                     hierarchy ? *MAT(hierarchy) : cv::noArray(), maxLevel, cv::Point((int)offset_x, (int)offset_y));
    return 0;
    CATCH_RET_INT
}

double opencv_imgproc_contour_area(void* contour, int oriented) {
    TRY return cv::contourArea(*MAT(contour), oriented != 0); CATCH_RET(0.0)
}

double opencv_imgproc_arc_length(void* curve, int closed) {
    TRY return cv::arcLength(*MAT(curve), closed != 0); CATCH_RET(0.0)
}

int opencv_imgproc_bounding_rect(void* points, int* x, int* y, int* width, int* height) {
    TRY
    cv::Rect r = cv::boundingRect(*MAT(points));
    *x = r.x; *y = r.y; *width = r.width; *height = r.height;
    return 0;
    CATCH_RET_INT
}

/* ============================================================
 * Imgproc - Histogram
 * ============================================================ */
int opencv_imgproc_equalize_hist(void* src, void* dst) {
    TRY cv::equalizeHist(*MAT(src), *MAT(dst)); return 0; CATCH_RET_INT
}

/* ============================================================
 * Imgproc - Additional feature detection (used by video4j)
 * ============================================================ */
int opencv_imgproc_corner_harris(void* src, void* dst, int blockSize, int ksize, double k, int borderType) {
    TRY cv::cornerHarris(*MAT(src), *MAT(dst), blockSize, ksize, k, borderType); return 0; CATCH_RET_INT
}

int opencv_imgproc_hough_lines(void* image, void* lines, double rho, double theta, int threshold, double srn, double stn) {
    TRY cv::HoughLines(*MAT(image), *MAT(lines), rho, theta, threshold, srn, stn); return 0; CATCH_RET_INT
}

int opencv_imgproc_hough_lines_p(void* image, void* lines, double rho, double theta, int threshold, double minLineLength, double maxLineGap) {
    TRY cv::HoughLinesP(*MAT(image), *MAT(lines), rho, theta, threshold, minLineLength, maxLineGap); return 0; CATCH_RET_INT
}

int opencv_imgproc_get_rect_sub_pix(void* image, double patchSize_w, double patchSize_h, double center_x, double center_y, void* patch, int patchType) {
    TRY cv::getRectSubPix(*MAT(image), cv::Size((int)patchSize_w, (int)patchSize_h), cv::Point2f((float)center_x, (float)center_y), *MAT(patch), patchType); return 0; CATCH_RET_INT
}

/* ============================================================
 * Core - Additional math (used by video4j)
 * ============================================================ */
int opencv_core_sqrt(void* src, void* dst) {
    TRY cv::sqrt(*MAT(src), *MAT(dst)); return 0; CATCH_RET_INT
}

/* ============================================================
 * VideoCapture (videoio module)
 * ============================================================ */
void* opencv_videocapture_create(void) {
    TRY return new cv::VideoCapture(); CATCH_RET_NULL
}

void opencv_videocapture_delete(void* cap) {
    delete CAP(cap);
}

int opencv_videocapture_open_file(void* cap, const char* filename, int apiPreference) {
    TRY return CAP(cap)->open(std::string(filename), apiPreference) ? 1 : 0; CATCH_RET_ZERO
}

int opencv_videocapture_open_device(void* cap, int index, int apiPreference) {
    TRY return CAP(cap)->open(index, apiPreference) ? 1 : 0; CATCH_RET_ZERO
}

int opencv_videocapture_is_opened(void* cap) {
    TRY return CAP(cap)->isOpened() ? 1 : 0; CATCH_RET_ZERO
}

void opencv_videocapture_release(void* cap) {
    if (cap) CAP(cap)->release();
}

int opencv_videocapture_read(void* cap, void* image) {
    TRY return CAP(cap)->read(*MAT(image)) ? 1 : 0; CATCH_RET_ZERO
}

int opencv_videocapture_grab(void* cap) {
    TRY return CAP(cap)->grab() ? 1 : 0; CATCH_RET_ZERO
}

int opencv_videocapture_retrieve(void* cap, void* image, int flag) {
    TRY return CAP(cap)->retrieve(*MAT(image), flag) ? 1 : 0; CATCH_RET_ZERO
}

int opencv_videocapture_set(void* cap, int propId, double value) {
    TRY return CAP(cap)->set(propId, value) ? 1 : 0; CATCH_RET_ZERO
}

double opencv_videocapture_get(void* cap, int propId) {
    TRY return CAP(cap)->get(propId); CATCH_RET(0.0)
}

} // extern "C"
