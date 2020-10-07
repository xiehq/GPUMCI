#include <GPUMCI/utils/CudaCosWeighting.h>
#include <odl_cpp_utils/cuda/cuda_utils.h>
#include <odl_cpp_utils/cuda/CuVector.h>

using namespace Eigen;

namespace gpumci {
namespace cuda {
extern void apply_cosweighting(const float3 sourcePosition,
                               const float3 detectorOrigin,
                               const float3 pixelDirectionU,
                               const float3 pixelDirectionV,
                               const int2 detectorSize,
                               float* source,
                               float* projection);

} //cuda

CudaCosWeighting::CudaCosWeighting(Vector2i detectorSize)
    : _detectorSize(detectorSize) {
}

void CudaCosWeighting::apply(Vector3d sourcePosition,
                             Vector3d detectorOrigin,
                             Vector3d pixelDirectionU,
                             Vector3d pixelDirectionV,
                             float* source,
                             float* target) {
    assert(pixelDirectionU.norm() > 0.0); //Pixel size cant be zero
    assert(pixelDirectionV.norm() > 0.0); //Pixel size cant be zero

    cuda::apply_cosweighting(make_float3(sourcePosition),
                             make_float3(detectorOrigin),
                             make_float3(pixelDirectionU),
                             make_float3(pixelDirectionV),
                             make_int2(_detectorSize),
                             source,
                             target);
}
} //gpumci
