cmake_minimum_required(VERSION 3.24)
project(test_yolo)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Development Interpreter)
find_package(OpenCV 4 REQUIRED)

include_directories(${Python_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

pybind11_add_module(_process SHARED process.cpp)
target_link_libraries(_process PRIVATE ${OpenCV_LIBS})
