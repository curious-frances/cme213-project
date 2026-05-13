# ============================================================================
# Makefile — CME 213 Final Project  (CPU + GPU stereo SAD)
# ============================================================================

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -march=native

NVCC       = nvcc
NVCCFLAGS  = -std=c++17 -O2 -arch=sm_75   # sm_75 = Turing (gpu-turing partition)

# ---------------------------------------------------------------------------
# Google Test
# ---------------------------------------------------------------------------
GTEST_ROOT = ./googletest-main
GTEST_DIR  = $(GTEST_ROOT)/googletest
GTEST_INC  = $(GTEST_DIR)/include
GTEST_SRC  = $(GTEST_DIR)/src

GTEST_FLAGS   = -isystem $(GTEST_INC) -O2 -std=c++17
GTEST_HEADERS = $(GTEST_INC)/gtest/*.h \
                $(GTEST_INC)/gtest/internal/*.h
GTEST_SRCS_   = $(GTEST_SRC)/*.cc $(GTEST_SRC)/*.h $(GTEST_HEADERS)

CPPFLAGS = -isystem $(GTEST_INC)

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
PERCEPTION_SRC = perception_cpu.cpp
PERCEPTION_HDR = perception_cpu.h perception_common.h

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
default: main_cpu main_gpu

# ---------------------------------------------------------------------------
# CPU objects
# ---------------------------------------------------------------------------
perception_cpu.o: $(PERCEPTION_SRC) $(PERCEPTION_HDR)
	$(CXX) $(CXXFLAGS) -c $(PERCEPTION_SRC) -o $@

main_cpu.o: main_cpu.cpp $(PERCEPTION_HDR)
	$(CXX) $(CXXFLAGS) -c main_cpu.cpp -o $@

main_cpu: main_cpu.o perception_cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# ---------------------------------------------------------------------------
# GPU objects
# ---------------------------------------------------------------------------
perception_gpu.o: perception_gpu.cu perception_gpu.h $(PERCEPTION_HDR)
	$(NVCC) $(NVCCFLAGS) -c perception_gpu.cu -o $@

main_gpu.o: main_gpu.cu perception_gpu.h $(PERCEPTION_HDR)
	$(NVCC) $(NVCCFLAGS) -c main_gpu.cu -o $@

main_gpu: main_gpu.o perception_gpu.o perception_cpu.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# ---------------------------------------------------------------------------
# GoogleTest
# ---------------------------------------------------------------------------
gtest: gtest.a gtest_main.a

gtest-all.o: $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $(GTEST_SRC)/gtest-all.cc -o $@

gtest_main.o: $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $(GTEST_SRC)/gtest_main.cc -o $@

gtest.a: gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a: gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# ---------------------------------------------------------------------------
# Test binary
# ---------------------------------------------------------------------------
perception_cpu_test.o: perception_cpu_test.cpp $(PERCEPTION_HDR) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c perception_cpu_test.cpp -o $@

test_cpu: perception_cpu_test.o perception_cpu.o gtest_main.a
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread
	./test_cpu

# ---------------------------------------------------------------------------
# Convenience targets
# ---------------------------------------------------------------------------
run: main_cpu
	./main_cpu --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 5

run_gpu: main_gpu
	./main_gpu --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 5

bench: main_cpu main_gpu
	./main_cpu --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 10 --csv benchmark.csv
	./main_gpu --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 10 --no-cpu --csv benchmark.csv

clean:
	rm -f main_cpu main_gpu test_cpu *.o *.a *.pgm *.csv
