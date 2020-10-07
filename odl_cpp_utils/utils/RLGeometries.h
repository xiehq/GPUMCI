#pragma once

#include <forwardprojection/IForwardProjector.h>
#include <Eigen/StdVector>
#define EIGEN_USE_NEW_STDVECTOR

namespace SimRec2D {
struct FanBeamGeometry {
    const Eigen::Vector2d sourcePosition;
    const Eigen::Vector2d detectorOrigin;
    const Eigen::Vector2d pixelDirection;

    void setGeometry(const int index,
                     Eigen::Vector2d& pixelPositionOut,
                     Eigen::Vector2d& directionOut) const {
        pixelPositionOut = detectorOrigin + pixelDirection * static_cast<double>(index);
        directionOut = (sourcePosition - pixelPositionOut).normalized();
    }
};

struct ParallelBeamGeometry {
    ParallelBeamGeometry(Eigen::Vector2d detectorOrigin,
                         Eigen::Vector2d pixelDirection)
        : _detectorOrigin(detectorOrigin),
          _pixelDirection(pixelDirection),
          _direction(Eigen::Vector2d(-pixelDirection[1], pixelDirection[0]).normalized()) {
    }

    void setGeometry(const int index,
                     Eigen::Vector2d& pixelPositionOut,
                     Eigen::Vector2d& directionOut) const {
        pixelPositionOut = _detectorOrigin + _pixelDirection * static_cast<double>(index);
        directionOut = _direction;
    }

  private:
    const Eigen::Vector2d _detectorOrigin;
    const Eigen::Vector2d _pixelDirection;
    const Eigen::Vector2d _direction;
};

struct LookupTableGeometry {
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pixelPositions;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> directions;

    void setGeometry(const int index,
                     Eigen::Vector2d& pixelPositionOut,
                     Eigen::Vector2d& directionOut) const {
        pixelPositionOut = pixelPositions[index];
        directionOut = directions[index];
    }
};

struct UnevenParallelBeamGeometry {
    UnevenParallelBeamGeometry(Eigen::Vector2d detectorOrigin,
                               Eigen::Vector2d pixelDirection,
                               size_t detectorSize)
        : _detectorOrigin(detectorOrigin),
          _pixelDirection(pixelDirection),
          _direction(Eigen::Vector2d(-pixelDirection[1], pixelDirection[0]).normalized()),
          _detectorSize(detectorSize) {
    }

    void setGeometry(const int index,
                     Eigen::Vector2d& pixelPositionOut,
                     Eigen::Vector2d& directionOut) const {
        double x = static_cast<double>(index) / _detectorSize * M_PI;
        double y = (sin(x * 2.0) + (x * 2)) * _detectorSize / (2 * M_PI);

        pixelPositionOut = _detectorOrigin + _pixelDirection * y;
        directionOut = _direction;
    }

  private:
    const Eigen::Vector2d _detectorOrigin;
    const Eigen::Vector2d _pixelDirection;
    const Eigen::Vector2d _direction;
    const double _detectorSize;
};
}
