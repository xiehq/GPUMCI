#include "CudaQuaternion.h"
#include <vector>

namespace gpumci {
namespace cuda {

__global__ void cudaRotationKernel(cuda::CudaMonteCarloParticle* particles, int numParticles, float angle, float3 rotationAxis) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numParticles) return;
    Quaternion rotation = Quaternion::describe_rotation(rotationAxis, angle);
    particles[idx].position = rotation.rotate(particles[idx].position);
    particles[idx].direction = rotation.rotate(particles[idx].direction);
}
void rotateParticles(std::vector<cuda::CudaMonteCarloParticle>& particles, float3& axis, float angle) {
    cuda::CudaMonteCarloParticle* d_particles;
    size_t numParticles = particles.size();

    cudaMalloc(&d_particles, numParticles * sizeof(cuda::CudaMonteCarloParticle));
    cudaMemcpy(d_particles, particles.data(), numParticles * sizeof(cuda::CudaMonteCarloParticle), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    int blockSize = 256; // This could be optimized based on the architecture
    int gridSize = (numParticles + blockSize - 1) / blockSize;

    // Quaternion::
    cudaRotationKernel<<<gridSize, blockSize>>>(d_particles, numParticles, angle, axis);

    cudaMemcpy(particles.data(), d_particles, numParticles * sizeof(cuda::CudaMonteCarloParticle), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
}

} // namespace cuda
} // namespace gpumci
