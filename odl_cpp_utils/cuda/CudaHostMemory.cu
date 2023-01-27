#include <odl_cpp_utils/cuda/CudaHostMemory.h>
#include <odl_cpp_utils/cuda/errcheck.h>
#include <stdint.h>

float *allocate_host_memory(unsigned N)
{

    float *p;
    cudaMallocHost((void**)&p, sizeof(float)*N);
    return p;
}

void free_host_mem(float* k)
{

    //free memory
    cudaFreeHost(k);
} 
