cmake_minimum_required(VERSION 3.24)
project(test_yolo)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Development Interpreter)

include_directories(${Python_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})

add_executable(test_yolo process.cpp)
target_link_libraries(test_yolo pybind11::pybind11 Python::Python)
