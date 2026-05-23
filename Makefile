.PHONY: all build clean

ORT_DIR  ?= /usr/local/onnxruntime
TORCH_DIR ?= /usr/local/libtorch
BUILD_DIR := build

all: build

$(BUILD_DIR)/backend_profiler: CMakeLists.txt $(wildcard src/*.cpp) $(wildcard include/*.h)
	cmake -B $(BUILD_DIR) \
		-DONNXRUNTIME_DIR=$(ORT_DIR) \
		-DLIBTORCH_DIR=$(TORCH_DIR)
	cmake --build $(BUILD_DIR) -j

build: $(BUILD_DIR)/backend_profiler

clean:
	[ -d $(BUILD_DIR) ] && rm -rf $(BUILD_DIR) || true
	(cd Plotting && make clean)