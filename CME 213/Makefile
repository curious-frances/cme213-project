# ============================================================================
# Makefile — CME 213 Final Project  (CPU baseline, Milestone 3)
# Mirrors the structure of the homework Makefile.
#
# Requirements:
#   - GCC or Clang with C++17 support
#   - GoogleTest source tree at ./googletest-main  (same path as the homework)
#
# Targets:
#   make              → build main_cpu
#   make test_cpu     → build and run GoogleTest suite
#   make run          → build and run main_cpu with default arguments
#   make clean        → remove all build artifacts
# ============================================================================

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -march=native

# ---------------------------------------------------------------------------
# Google Test — same directory convention as the homework Makefile
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
# Default target: build the main driver
# ---------------------------------------------------------------------------
default: main_cpu

# ---------------------------------------------------------------------------
# Object files
# ---------------------------------------------------------------------------
perception_cpu.o: $(PERCEPTION_SRC) $(PERCEPTION_HDR)
	$(CXX) $(CXXFLAGS) -c $(PERCEPTION_SRC) -o $@

main_cpu.o: main_cpu.cpp $(PERCEPTION_HDR)
	$(CXX) $(CXXFLAGS) -c main_cpu.cpp -o $@

# ---------------------------------------------------------------------------
# Main driver binary
# ---------------------------------------------------------------------------
main_cpu: main_cpu.o perception_cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# ---------------------------------------------------------------------------
# GoogleTest infrastructure — identical to the homework Makefile
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

# Run a quick small case for fast iteration during development
run_small: main_cpu
	./main_cpu --height 120 --width 160 --disp 8 --max-disp 32 --radius 2 --repeats 3

clean:
	rm -f main_cpu test_cpu *.o *.a
