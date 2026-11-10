cmake_minimum_required(VERSION 3.16)
project(HEScan CXX)

# -DHESCAN_GPU=ON adds the FIDESlib (CUDA) backend: hescan_demo --gpu 1.
option(HESCAN_GPU "Build the FIDESlib GPU backend (needs CUDA 12/13, FIDESlib, patched OpenFHE 1.6.1)" OFF)

if(HESCAN_GPU)
    set(CMAKE_CXX_STANDARD 20)  # FIDESlib headers use <source_location>, concepts
else()
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(HESCAN_GPU)
    # FIDESlib embeds a patched OpenFHE 1.6.1 as static archives. Compile against exactly that
    # install (pass -DOpenFHE_DIR=<prefix>/lib/OpenFHE) and link its static libraries, never a
    # second (shared) OpenFHE: two copies in one process break OpenFHE's global key maps.
    find_package(OpenFHE 1.6.1 EXACT REQUIRED CONFIG)
    set(BUILD_STATIC ON)
    find_package(fideslib REQUIRED CONFIG)  # -Dfideslib_DIR=<prefix>/share/fideslib/cmake
    find_package(CUDAToolkit REQUIRED)
    # FIDESlib's headers change struct layouts under -DNCCL, and FIDESlib turns that on iff it
    # finds libnccl at build time. Mirror its detection so both sides agree; override with
    # -DHESCAN_FIDESLIB_NCCL=ON/OFF if FIDESlib was built on a different machine.
    if(NOT DEFINED HESCAN_FIDESLIB_NCCL)
        find_library(HESCAN_NCCL_LIB NAMES nccl HINTS /usr/local/cuda/lib64 /usr/lib /usr/local/lib)
        if(HESCAN_NCCL_LIB)
            set(HESCAN_FIDESLIB_NCCL ON)
        else()
            set(HESCAN_FIDESLIB_NCCL OFF)
        endif()
    endif()
    message(STATUS "HEScan GPU backend: FIDESlib ${fideslib_VERSION}, CUDA ${CUDAToolkit_VERSION}, NCCL macros ${HESCAN_FIDESLIB_NCCL}")
else()
    # Complex packing (CKKSDataType::COMPLEX) needs OpenFHE >= 1.4.
    find_package(OpenFHE 1.4 REQUIRED)
endif()

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

if(HESCAN_GPU)
    add_library(hescan_gpu STATIC hescan_gpu.cpp)
    target_link_libraries(hescan_gpu PUBLIC hescan fideslib::fideslib CUDA::cudart CUDA::cuda_driver ${OFHE_LIBS})
    if(HESCAN_FIDESLIB_NCCL)
        target_compile_definitions(hescan_gpu PRIVATE MULTI_GPU NCCL RCCL)
    endif()
    target_compile_definitions(hescan_demo PRIVATE HESCAN_WITH_GPU)
    target_link_libraries(hescan_demo hescan_gpu)
endif()
