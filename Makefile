# CUDA paths (adjust if needed)
CUDA_PATH := /usr/local/cuda-12.2/targets/x86_64-linux
NVCC := nvcc

# Files
PTX_TARGET := face_verify.ptx
CU_SRC := face_verify.cu

CPP_TARGET := run_ptx
CPP_SRC := test_batch.cpp

# Flags
CPP_INCLUDES := -I$(CUDA_PATH)/include
CPP_LIBS := -L$(CUDA_PATH)/lib -lcuda

# Default build
all: $(PTX_TARGET) $(CPP_TARGET)

# Compile .cu to .ptx
$(PTX_TARGET): $(CU_SRC)
	$(NVCC) -ptx $< -o $@

# Compile host .cpp
$(CPP_TARGET): $(CPP_SRC)
	g++ $< -o $@ $(CPP_INCLUDES) $(CPP_LIBS)

clean:
	rm -f $(PTX_TARGET) $(CPP_TARGET)