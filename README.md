# OpenCV-FFM

OpenCV-FFM provides modern Java bindings for OpenCV 4.x using the Foreign Function & Memory (FFM) API (Project Panama). It enables high-performance, zero-JNI access to OpenCV's core, imgproc, and videoio modules directly from Java.

## Features

- Zero-JNI, pure FFM (Project Panama) bindings
- Supports OpenCV 4.10.0 (core, imgproc, videoio, imgcodecs)
- Native resource extraction and loading (no manual LD_LIBRARY_PATH needed)
- Full Java 25+ support
- Designed for integration with [video4j](https://github.com/metaloom/video4j)

## Limitations

- Currently only AMD64 Linux is supported
- Other platforms may be added in the future

## Usage

Add to your Maven project:

```xml
<dependency>
  <groupId>io.metaloom.opencv</groupId>
  <artifactId>opencv-ffm</artifactId>
  <version>4.10.0-SNAPSHOT</version>
</dependency>
```

### Example: Load and Use OpenCV

```java
import io.metaloom.opencv.OpenCVLoader;
import io.metaloom.opencv.core.Mat;
import io.metaloom.opencv.imgproc.Imgproc;

// Load the native library (auto-extracts if needed)
OpenCVLoader.load();

// Create a Mat and perform operations
try (Mat mat = new Mat(100, 100, CvType.CV_8UC3)) {
    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
    // ...
}
```

## Native Library

The native library (`libopencv_ffm.so`) is bundled in the JAR under `native/linux/` and is extracted automatically at runtime. No manual setup is required.

## Build

### Requirements
- OpenCV 4.10.0 (built with FFMPEG support)
- JDK 25 or newer
- Maven
- GCC 13+
- CMake 3.16+

### Build Steps

```bash
# Build native library and Java bindings
./build.sh
```

This will:
- Build the native shared library (`libopencv_ffm.so`)
- Copy it to the JAR resources
- Build and install the Maven artifact

## License

Apache License 2.0. See LICENSE.txt for details.
