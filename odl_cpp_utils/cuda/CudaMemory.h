#include <stddef.h>

/*
 * A very simple class for holding CUDA memory.
 *
 * Template parameter can be float or uint8_t
 */
template <typename T>
class CudaMemory {
  public:
    CudaMemory(size_t size);

    ~CudaMemory();

    T* device_ptr();
    const T* device_ptr() const;

    void copy_from_host(const T* data_ptr_host);
    void copy_to_host(T* data_ptr_host) const;

  private:
    size_t _size;
    T* _data_ptr_device;
};
