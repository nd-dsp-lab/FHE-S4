# Build for STEP 1 (fhe/measure_epoch.cpp). Standalone on purpose: the top-level
# CMakeLists.txt builds every OpenFHE demo in the repo, and none of them is needed
# to measure the epoch.
#
# Follows OpenFHE's own CMakeLists.User.txt rather than hard-coding
# /usr/local/include/openfhe, so the SAME file builds against the workstation's
# system install and against a private install:
#
#   cmake -S fhe -B fhe/build -DOpenFHE_DIR=/usr/local/lib/OpenFHE
#   cmake --build fhe/build -j
#
# OpenFHE_CXX_FLAGS matters: it carries the flags the library itself was built
# with (OpenMP, the math backend, NATIVEINT). Compiling user code with different
# ones is a source of silent mismatches.

cmake_minimum_required(VERSION 3.16)
project(FHE_Epoch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenFHE CONFIG REQUIRED)
message(STATUS "OpenFHE version:       ${BASE_OPENFHE_VERSION}")
message(STATUS "OpenFHE include:       ${OpenFHE_INCLUDE}")
message(STATUS "OpenFHE native size:   ${OpenFHE_NATIVE_SIZE}")

set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -O3")

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

if(UNIX AND NOT APPLE)
    add_link_options(-Wl,--no-as-needed)
endif()
link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
link_libraries(${OpenFHE_SHARED_LIBRARIES})

add_executable(measure_epoch measure_epoch.cpp)

# Record the library version next to the binary, so the sweep can put it in
# every result row without guessing.
file(WRITE ${CMAKE_BINARY_DIR}/openfhe_version.txt "${BASE_OPENFHE_VERSION}\n")
