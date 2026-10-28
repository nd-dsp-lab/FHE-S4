cmake_minimum_required(VERSION 3.16)
project(seglut_invsqrt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# links against the system-wide OpenFHE on tjws-03 (/usr/local/lib/OpenFHE); installs nothing
find_package(OpenFHE REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS}")

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

add_executable(bench bench.cpp)
target_link_libraries(bench ${OpenFHE_SHARED_LIBRARIES})
