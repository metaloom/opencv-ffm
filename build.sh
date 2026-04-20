#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Building native library ==="
bash "${SCRIPT_DIR}/native/build.sh" "$@"

echo ""
echo "=== Copying native library to resources ==="
RESOURCES_DIR="${SCRIPT_DIR}/src/main/resources/native/linux"
mkdir -p "${RESOURCES_DIR}"
cp "${SCRIPT_DIR}/native/build/libopencv_ffm.so.1.0.0" "${RESOURCES_DIR}/libopencv_ffm.so"
echo "Copied to ${RESOURCES_DIR}/libopencv_ffm.so"

echo ""
echo "=== Building Maven project ==="
cd "${SCRIPT_DIR}"
mvn install -DskipTests -q

echo ""
echo "Build complete."
