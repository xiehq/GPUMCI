#pragma once

#include <cstdio>

#define CUDA_SAFE_CALL(ans)                         \
    do {                                            \
        gpuAssert((ans), __FILE__, __LINE__, #ans); \
    } while (0)

#define CUDA_KERNEL_ERRCHECK                                                               \
    do {                                                                                   \
        gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__, "Kernel call");               \
        gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__, "Kernel call, syncronize"); \
    } while (0)

inline void gpuAssert(cudaError_t code, const char* file, int line, const char* call, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR ENCOUNTERED:\n ERROR: %s\n FILE: %s\n LINE: %d\n CALL: %s\n", cudaGetErrorString(code), file, line, call);
        if (abort) exit(code);
    }
}
