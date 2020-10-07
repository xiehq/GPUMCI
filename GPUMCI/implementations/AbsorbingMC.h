#include <Eigen/Core>
#include <stdint.h>

namespace gpumci {
class AbsorbingMC {
  public:
    AbsorbingMC(const Eigen::Vector3i& volumeSize,
                const Eigen::Vector3d& volumeOrigin,
                const Eigen::Vector3d& voxelSize,
                const Eigen::Vector2i& detectorSize);

    void setData(const float* densityDevice,
                 const uint8_t* materialTypeDevice);

    void project(const Eigen::Vector3d& sourcePosition,
                 const Eigen::Vector3d& detectorOrigin,
                 const Eigen::Vector3d& pixelDirectionU,
                 const Eigen::Vector3d& pixelDirectionV,
                 float* target) const;

  private:
    const Eigen::Vector3i _volumeSize;
    const Eigen::Vector3d _volumeOrigin;
    const Eigen::Vector3d _voxelSize;
    const Eigen::Vector2i _detectorSize;
    const float* _densityDevice;
    const uint8_t* _materialTypeDevice;
};
}
