#include <iostream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

namespace py = pybind11;

void dataprocess(std::vector<double> box, std::vector<double> mask) {
    // TODO: 处理识别到的数据
    // box: 长度为6的数组 [x, y, x, y, 置信度， 类别]
    // mask: 长度不定， 形式为 [[点1坐标]， [点2坐标]， ...]
}

PYBIND11_MODULE(_process, m) {
    m.def("dataprocess", &dataprocess);
}

