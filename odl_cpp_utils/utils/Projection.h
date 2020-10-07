#pragma once

#include <Eigen/Core>

namespace SimRec2D {
/**
 * Definition of an projection image with geometry data in two dimensions.
 */
struct Projection {
    Eigen::Vector2d sourcePosition;
    Eigen::Vector2d detectorOrigin;
    Eigen::Vector2d pixelDirection;
    Eigen::ArrayXd projection;

    Projection(const Eigen::Vector2d& _sourcePosition,
               const Eigen::Vector2d& _detectorOrigin,
               const Eigen::Vector2d& _pixelDirection,
               const Eigen::ArrayXd& _projection)
        : sourcePosition(_sourcePosition),
          detectorOrigin(_detectorOrigin),
          pixelDirection(_pixelDirection),
          projection(_projection) {
    }
};
}