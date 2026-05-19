cmake_minimum_required(VERSION 3.12)
project(inference_profiler)
set(CMAKE_CXX_STANDARD 20)
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation")
set(CLI11_DIR "${CMAKE_SOURCE_DIR}/../external/CLI11" CACHE PATH "Path to CLI11 installation")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Works with both .so and .so.1
find_library(ORT_LIB 
    NAMES libonnxruntime.so.1 libonnxruntime.so onnxruntime
    PATHS ${ONNXRUNTIME_DIR}/runtimes/linux-x64/native
    NO_DEFAULT_PATH
)
if(NOT ORT_LIB)
    message(FATAL_ERROR "Could not find libonnxruntime in ${ONNXRUNTIME_DIR}/runtimes/linux-x64/native")
endif()
message(STATUS "Found ORT lib: ${ORT_LIB}")

find_path(CLI11_INCLUDE_DIR
    NAMES CLI/CLI.hpp
    PATHS ${CLI11_DIR}/include
    NO_DEFAULT_PATH
)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(inference_profiler ${SOURCES})
target_include_directories(inference_profiler PRIVATE
    ${ONNXRUNTIME_DIR}/build/native/include
    ${CLI11_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(inference_profiler PRIVATE ${ORT_LIB})