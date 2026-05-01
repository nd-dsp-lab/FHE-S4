cmake_minimum_required(VERSION 3.16)
project(OpenFHE_Toeplitz)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(OpenFHE REQUIRED)

add_executable(openfhe_toeplitz openfhe_toeplitz.cpp)
add_executable(openfhe_s4 openfhe_s4.cpp)
add_executable(eval_chebyshev_series EvalChebyshevSeries_demo.cpp)
add_executable(openfhe_rotate_sum_protocol openfhe_rotate_sum_protocol.cpp)
add_executable(openfhe_dense_toeplitz_baseline openfhe_dense_toeplitz_baseline.cpp)

target_include_directories(openfhe_toeplitz PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)
target_include_directories(openfhe_s4 PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)
target_include_directories(eval_chebyshev_series PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)
target_include_directories(openfhe_rotate_sum_protocol PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)
target_include_directories(openfhe_dense_toeplitz_baseline PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)

target_compile_options(openfhe_toeplitz PRIVATE -fopenmp)
target_compile_options(openfhe_s4 PRIVATE -fopenmp)
target_compile_options(eval_chebyshev_series PRIVATE -fopenmp)
target_compile_options(openfhe_rotate_sum_protocol PRIVATE -fopenmp)
target_compile_options(openfhe_dense_toeplitz_baseline PRIVATE -fopenmp)
target_link_libraries(openfhe_toeplitz OPENFHEpke OPENFHEcore OPENFHEbinfhe -fopenmp)
target_link_libraries(openfhe_s4 OPENFHEpke OPENFHEcore OPENFHEbinfhe nlohmann_json::nlohmann_json -fopenmp)
target_link_libraries(eval_chebyshev_series OPENFHEpke OPENFHEcore OPENFHEbinfhe nlohmann_json::nlohmann_json -fopenmp)
target_link_libraries(openfhe_rotate_sum_protocol OPENFHEpke OPENFHEcore OPENFHEbinfhe -fopenmp)
target_link_libraries(openfhe_dense_toeplitz_baseline OPENFHEpke OPENFHEcore OPENFHEbinfhe -fopenmp)