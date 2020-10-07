#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/NonLinearOptimization>

namespace SimRec2D {
// Generic functor
template <typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor {
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
};

template <typename Functor, typename... Args>
Eigen::NumericalDiff<Functor> makeNumbericalDiff(Args&&... args) {
    return Eigen::NumericalDiff<Functor>(Functor(std::forward<Args>(args)...));
}

template <typename Functor>
Eigen::LevenbergMarquardt<Functor> makeLevenbergMarquardt(Functor& functor) {
    return Eigen::LevenbergMarquardt<Functor>(functor);
}
}