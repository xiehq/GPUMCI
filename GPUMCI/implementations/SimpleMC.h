#include <memory>
#include <Eigen/Core>
#include <stdint.h>
#include <GPUMCI/physics/CudaSettings.h>

namespace gpumci {
namespace cuda {
struct SimpleMCCuData;
}
class SimpleMC {
  public:
    SimpleMC(const Eigen::Vector3i& volumeSize,
             const Eigen::Vector3d& volumeOrigin,
             const Eigen::Vector3d& voxelSize,
             const Eigen::Vector2i& detectorSize);

    void setData(const float* densityDevice,
                 const uint8_t* materialTypeDevice);

    void project(const Eigen::Vector3d& sourcePosition,
                 const Eigen::Vector3d& detectorOrigin,
                 const Eigen::Vector3d& pixelDirectionU,
                 const Eigen::Vector3d& pixelDirectionV,
                 float* primary,
                 float* scatter) const;

  private:
    cuda::CudaParameters _param;
    const Eigen::Vector3i _volumeSize;
    const Eigen::Vector3d _volumeOrigin;
    const Eigen::Vector3d _voxelSize;
    const Eigen::Vector2i _detectorSize;
    std::shared_ptr<cuda::SimpleMCCuData> _cudaData;
};
}
