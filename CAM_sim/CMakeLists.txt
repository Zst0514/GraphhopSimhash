cmake_minimum_required(VERSION 3.16)
project(graphhop_hash_reuse_hw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ghhw_common
    common/trace_loader.cpp
    common/report_writer.cpp
    common/progress.cpp
)
target_include_directories(ghhw_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(digital_logic_cpp)
add_subdirectory(analog_cam_cpp)

enable_testing()
add_executable(hardware_model_tests
    tests/hardware_model_tests.cpp
)
target_link_libraries(hardware_model_tests PRIVATE ghhw_common digital_engine analog_cam_engine)
add_test(NAME hardware_model_tests COMMAND hardware_model_tests)
