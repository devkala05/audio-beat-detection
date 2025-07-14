TARGET = beat_detect
SRCS = src/main.cu src/beat_kernel.cu
INCLUDES = -Iinclude
LIBS = -lsndfile
NVCC_FLAGS = -std=c++17 -O2

all:
	nvcc $(NVCC_FLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
