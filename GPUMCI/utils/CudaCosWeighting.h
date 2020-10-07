#pragma once

#include <memory>
#include <Eigen/Dense>

namespace gpumci {
/**
 * Divides each pixel value by the cosine of the incoming angle
 */
class CudaCosWeighting {
  public:
    CudaCosWeighting(Eigen::Vector2i detectorSize);

    void apply(Eigen::Vector3d sourcePosition,
               Eigen::Vector3d detectorOrigin,
               Eigen::Vector3d pixelDirectionU,
               Eigen::Vector3d pixelDirectionV,
               float* source,
               float* target);

  private:
    const Eigen::Vector2i _detectorSize;
};
}
