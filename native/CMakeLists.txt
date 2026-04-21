cmake_minimum_required(VERSION 3.16)
project(opencv_ffm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV – set OpenCV_DIR via cmake -DOpenCV_DIR=... or environment
if(DEFINED ENV{OpenCV_DIR} AND NOT OpenCV_DIR)
    set(OpenCV_DIR $ENV{OpenCV_DIR} CACHE PATH "Path to OpenCV build directory")
endif()

if(OpenCV_DIR)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio objdetect PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio objdetect)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_library(opencv_ffm SHARED opencv_ffm.cpp)

target_include_directories(opencv_ffm PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_ffm PRIVATE ${OpenCV_LIBS})

# Set SONAME
set_target_properties(opencv_ffm PROPERTIES
    SOVERSION 1
    VERSION 1.0.0
    OUTPUT_NAME opencv_ffm
)
