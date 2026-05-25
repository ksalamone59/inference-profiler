.PHONY: all build clean

ORT_DIR  ?= /usr/local/onnxruntime
TORCH_DIR ?= /usr/local/libtorch
BUILD_DIR := build
TEST_BUILD_DIR := build_test

all: build test

$(BUILD_DIR)/backend_profiler: CMakeLists.txt $(wildcard src/*.cpp) $(wildcard include/*.h)
	cmake -B $(BUILD_DIR) \
		-DONNXRUNTIME_DIR=$(ORT_DIR) \
		-DLIBTORCH_DIR=$(TORCH_DIR)
	cmake --build $(BUILD_DIR) -j

$(TEST_BUILD_DIR)/backend_ci_testing: CMakeLists.txt cpp/src/ci_testing.cpp cpp/include/median.h cpp/include/metrics.h cpp/include/data_provider.h
	cmake -B $(TEST_BUILD_DIR) \
		-DDO_TESTS=ON
	cmake --build $(TEST_BUILD_DIR) -j
	(cd $(TEST_BUILD_DIR) && ./backend_ci_testing)

build: $(BUILD_DIR)/backend_profiler

test: $(TEST_BUILD_DIR)/backend_ci_testing

clean:
	[ -d $(BUILD_DIR) ] && rm -rf $(BUILD_DIR) || true
	[ -d $(TEST_BUILD_DIR) ] && rm -rf $(TEST_BUILD_DIR) || true
	(cd Plotting && make clean)