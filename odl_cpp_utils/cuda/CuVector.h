#pragma once

#include <Eigen/Core>
#include "cutil_math.h"

template <typename T>
inline float2 make_float2(const Eigen::Matrix<T, 2, 1>& vec) {
    return {static_cast<float>(vec[0]), static_cast<float>(vec[1])};
}

template <typename T>
inline float3 make_float3(const Eigen::Matrix<T, 3, 1>& vec) {
    return {static_cast<float>(vec[0]), static_cast<float>(vec[1]), static_cast<float>(vec[2])};
}

template <typename T>
inline int2 make_int2(const Eigen::Matrix<T, 2, 1>& vec) {
    return {vec[0], vec[1]};
}

template <typename T>
inline int3 make_int3(const Eigen::Matrix<T, 3, 1>& vec) {
    return {vec[0], vec[1], vec[2]};
}