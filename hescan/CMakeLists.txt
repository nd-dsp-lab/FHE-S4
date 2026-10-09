cmake_minimum_required(VERSION 3.16)
project(HEScan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Complex packing (CKKSDataType::COMPLEX) needs OpenFHE >= 1.4.
find_package(OpenFHE 1.4 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS}")
include_directories(${OPENMP_INCLUDES} ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
                    ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
link_directories(${OpenFHE_LIBDIR} ${OPENMP_LIBRARIES})
# find the OpenFHE shared libs at run time without LD_LIBRARY_PATH
set(CMAKE_BUILD_RPATH ${OpenFHE_LIBDIR})
# DT_RPATH (not RUNPATH) so it also covers libOPENFHEpke -> libOPENFHEbinfhe
add_link_options(-Wl,--disable-new-dtags)
if(BUILD_STATIC)
    set(OFHE_LIBS ${OpenFHE_STATIC_LIBRARIES})
else()
    set(OFHE_LIBS ${OpenFHE_SHARED_LIBRARIES})
endif()

add_library(hescan STATIC hescan.cpp)
target_link_libraries(hescan ${OFHE_LIBS})

add_executable(hescan_conj_check conj_check.cpp)
target_link_libraries(hescan_conj_check ${OFHE_LIBS})

add_executable(hescan_demo hescan_demo.cpp)
target_link_libraries(hescan_demo hescan ${OFHE_LIBS})
