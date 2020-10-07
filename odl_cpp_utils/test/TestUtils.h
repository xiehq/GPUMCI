#pragma once
#include <utils/PrintArray.h>

#include <string>
#include <iostream>
#include <Eigen/Core>

namespace SimRec2D {

/// If TEST_VERBOSE is set, print array info
template <typename T, int RowsAtCompileTime, int ColsAtCompileTime>
void testlog(const Eigen::Array<T, RowsAtCompileTime, ColsAtCompileTime>& data) {
#ifdef TEST_VERBOSE
    printArray(data);
#endif
}
}
