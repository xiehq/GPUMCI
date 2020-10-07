#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace SimRec2D {
/**
 * Calculates the pearson correlation coefficient, a measure of how well the arrays X and Y linearly approximate each other.
 *
 * Further info: http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
 */
template <typename Derived1, typename Derived2>
inline double correlationCoefficient(const Eigen::ArrayBase<Derived1>& X,
                                     const Eigen::ArrayBase<Derived2>& Y) {
    assert(X.cols() == Y.cols());
    assert(X.rows() == Y.rows());

    auto n = X.size();
    auto XSum = X.sum();
    auto YSum = Y.sum();

    auto t = n * (X * Y).sum() - XSum * YSum;
    auto s = std::sqrt(n * X.square().sum() - XSum * XSum) *
             std::sqrt(n * Y.square().sum() - YSum * YSum);

    return t / s;
}

/**
 * Calculates the coefficient of determination, a measure between 0 (no correlation) and 1 (perfect correlation)
 * Further info: http://en.wikipedia.org/wiki/Coefficient_of_determination
 */
template <typename Derived1, typename Derived2>
inline double calculateR2(const Eigen::ArrayBase<Derived1>& X,
                          const Eigen::ArrayBase<Derived2>& Y) {
    auto corrCoeff = correlationCoefficient(X, Y);
    return corrCoeff * corrCoeff;
}
}
