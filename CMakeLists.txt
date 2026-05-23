cmake_minimum_required(VERSION 3.12)
project(backend_profiler)
set(CMAKE_CXX_STANDARD 20)
set(LIBTORCH_DIR "" CACHE PATH "Path to Torch installation")
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation")
set(CLI11_DIR "${CMAKE_SOURCE_DIR}/external/CLI11" CACHE PATH "Path to CLI11 installation")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_ONNX "Use ONNX Runtime for inference" ON)
option(USE_TORCH "Use Torch for inference" ON)
if(USE_ONNX)
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
endif()

if(USE_TORCH)
    if(NOT LIBTORCH_DIR)
        message(FATAL_ERROR "USE_TORCH requires -DLIBTORCH_DIR=...")
    endif()
    set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
    find_package(Torch REQUIRED)
    message(STATUS "Found Torch: ${Torch_VERSION}")
endif()

find_path(CLI11_INCLUDE_DIR
    NAMES CLI/CLI.hpp
    PATHS ${CLI11_DIR}/include
    NO_DEFAULT_PATH
)

file(GLOB_RECURSE SOURCES "cpp/src/*.cpp")
add_executable(backend_profiler ${SOURCES})

target_include_directories(backend_profiler PRIVATE
    ${CLI11_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp/include
)

target_compile_options(backend_profiler PRIVATE -Wall -Wextra -Wpedantic)

if(USE_ONNX)
    target_include_directories(backend_profiler PRIVATE
        ${ONNXRUNTIME_DIR}/build/native/include
    )
    target_link_libraries(backend_profiler PRIVATE ${ORT_LIB})
    target_compile_definitions(backend_profiler PRIVATE USE_ONNX)
endif()

if(USE_TORCH)
    target_link_libraries(backend_profiler PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(backend_profiler PRIVATE USE_TORCH)
endif()