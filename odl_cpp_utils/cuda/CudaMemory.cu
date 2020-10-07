#include <odl_cpp_utils/cuda/CudaMemory.h>
#include <odl_cpp_utils/cuda/errcheck.h>
#include <stdint.h>

template <typename T>
CudaMemory<T>::CudaMemory(size_t size)
    : _size(size) {
    CUDA_SAFE_CALL(cudaMalloc(&_data_ptr_device, sizeof(T) * _size));
}

template <typename T>
CudaMemory<T>::~CudaMemory() {
    CUDA_SAFE_CALL(cudaFree(&_data_ptr_device));
}

template <typename T>
T* CudaMemory<T>::device_ptr() {
    return _data_ptr_device;
}

template <typename T>
const T* CudaMemory<T>::device_ptr() const {
    return _data_ptr_device;
}
template <typename T>
void CudaMemory<T>::copy_from_host(const T* data_ptr_host) {
    CUDA_SAFE_CALL(cudaMemcpy(_data_ptr_device, data_ptr_host, _size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void CudaMemory<T>::copy_to_host(T* data_ptr_host) const {
    CUDA_SAFE_CALL(cudaMemcpy(data_ptr_host, _data_ptr_device, _size * sizeof(T), cudaMemcpyDeviceToHost));
}

template class CudaMemory<float>;
template class CudaMemory<uint8_t>;
