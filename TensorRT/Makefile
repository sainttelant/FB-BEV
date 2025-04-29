# Define paths
CUDA_DIR ?= /usr/local/cuda
DLSW_TRIPLE = aarch64-linux-gnu
TRT_PATH ?= /usr/src/tensorrt/
TRT_LIB_DIR := $(TRT_PATH)/lib
TRT_INC_DIR := $(TRT_PATH)/include
SYSROOT ?= 
OUTPUT_DIR ?= build

# Compiler and flags
CXX := aarch64-linux-gnu-g++
NVCC := $(CUDA_DIR)/bin/nvcc

CXXFLAGS := -std=c++14 -O3 -Wno-deprecated-declarations -fPIC \
    -I$(TRT_INC_DIR)/ \
    -I$(CUDA_DIR)/include \
    -I$(SYSROOT)/include \
    -Icommon \

NVCCFLAGS := -std=c++14 --compiler-options '-fPIC' \
    -I$(TRT_INC_DIR)/ \
    -I$(CUDA_DIR)/include \
    -I$(SYSROOT)/include \
    -Icommon \
    -gencode arch=compute_87,code=sm_87

NVCCFLAGS += --compiler-bindir=/usr/bin/aarch64-linux-gnu-g++

# Library paths and libraries
LDFLAGS := -L$(TRT_LIB_DIR)  -L$(CUDA_DIR)/targets/aarch64-linux/lib
LDLIBS := -lnvinfer -lnvinfer_plugin -lcudart 

# Source and object files
CPP_SRCS := $(wildcard common/*.cpp plugin/bev_pool_v2/*.cpp plugin/grid_sampler/*.cpp)
CU_SRCS := $(wildcard common/*.cu plugin/bev_pool_v2/*.cu plugin/grid_sampler/*.cu)
CPP_OBJS := $(addprefix $(OUTPUT_DIR)/, $(CPP_SRCS:.cpp=.o))
CU_OBJS := $(addprefix $(OUTPUT_DIR)/, $(CU_SRCS:.cu=.o))

# Output shared library
OUTPUT_LIB := ./fb-occ_trt_plugin_aarch64.so

# Rules
all: $(OUTPUT_DIR) $(OUTPUT_LIB)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUTPUT_DIR)/%.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUTPUT_LIB): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(OUTPUT_DIR)
