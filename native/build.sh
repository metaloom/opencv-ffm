#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# OpenCV_DIR defaults to env var or can be passed as first argument
OPENCV_DIR="${1:-${OpenCV_DIR:-}}"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [ -n "${OPENCV_DIR}" ]; then
    CMAKE_ARGS+=(-DOpenCV_DIR="${OPENCV_DIR}")
fi

cmake "${CMAKE_ARGS[@]}" "${SCRIPT_DIR}"

make -j$(nproc)

echo ""
echo "Build complete. Library: ${BUILD_DIR}/libopencv_ffm.so"
echo "Exported symbols:"
nm -D "${BUILD_DIR}/libopencv_ffm.so" | grep " T opencv_" | head -20
echo "..."
nm -D "${BUILD_DIR}/libopencv_ffm.so" | grep -c " T opencv_"
echo "total exported opencv_ symbols"
