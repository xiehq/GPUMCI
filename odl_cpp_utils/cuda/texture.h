#pragma once

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <odl_cpp_utils/cuda/errcheck.h>

template <typename T>
struct BoundTexture1D {
  private:
    cudaArray_t _arr;
    cudaTextureObject_t _tex;
    bool has_array;
    bool has_data;

  public:
    size_t size;

    BoundTexture1D(const size_t size,
                   const cudaTextureAddressMode addressMode,
                   const cudaTextureFilterMode filterMode,
                   const cudaTextureReadMode readMode = cudaReadModeElementType)
        : size(size),
          has_array(false),
          has_data(false) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        CUDA_SAFE_CALL(cudaMallocArray(&_arr, &channelDesc, size));
        has_array = true;

        // create texture object
        cudaResourceDesc resourceDescriptor = {};
        resourceDescriptor.resType = cudaResourceTypeArray;
        resourceDescriptor.res.array.array = _arr;

        // Setup a texture descriptor
        cudaTextureDesc textureDescriptor = {};
        textureDescriptor.addressMode[0] = addressMode;
        textureDescriptor.filterMode = filterMode;
        textureDescriptor.readMode = readMode;
        textureDescriptor.normalizedCoords = 0;

        // Create the texture object
        CUDA_SAFE_CALL(cudaCreateTextureObject(&_tex, &resourceDescriptor, &textureDescriptor, NULL));
    }

    void setData(const T* source) {
        //CUDA_SAFE_CALL(cudaMemcpyToArray(_arr, 0, 0, source, size * sizeof(T), cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpy2DToArray(_arr, 0, 0, source, size * sizeof(T),  size * sizeof(T), 1, cudaMemcpyDeviceToDevice));
        has_data = true;
    }

    cudaTextureObject_t tex() const {
        if (!has_array) {
            throw std::runtime_error{"Texture array not set"};
        }
        if (!has_data) {
            throw std::runtime_error{"Texture data not set"};
        }
        return _tex;
    }

    ~BoundTexture1D() {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(_tex));
        if (has_array)
            CUDA_SAFE_CALL(cudaFreeArray(_arr));
    }

    BoundTexture1D& operator=(BoundTexture1D const&) = delete;
    BoundTexture1D(const BoundTexture1D& that) = delete;
};

template <typename T>
struct BoundTexture2D {
  private:
    cudaArray_t _arr;
    cudaTextureObject_t _tex;
    bool has_array;
    bool has_data;

  public:
    int2 size;

    BoundTexture2D(const int2 size,
                   const cudaTextureAddressMode addressMode,
                   const cudaTextureFilterMode filterMode,
                   const cudaTextureReadMode readMode = cudaReadModeElementType)
        : size(size),
          has_array(false),
          has_data(false) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        CUDA_SAFE_CALL(cudaMallocArray(&_arr, &channelDesc, size.x, size.y));
        has_array = true;

        // create texture object
        cudaResourceDesc resourceDescriptor = {};
        resourceDescriptor.resType = cudaResourceTypeArray;
        resourceDescriptor.res.array.array = _arr;

        // Setup a texture descriptor
        cudaTextureDesc textureDescriptor = {};
        textureDescriptor.addressMode[0] = addressMode;
        textureDescriptor.addressMode[1] = addressMode;
        textureDescriptor.filterMode = filterMode;
        textureDescriptor.readMode = readMode;
        textureDescriptor.normalizedCoords = 0;

        // Create the texture object
        CUDA_SAFE_CALL(cudaCreateTextureObject(&_tex, &resourceDescriptor, &textureDescriptor, NULL));
    }

    void setData(const T* source) {
        CUDA_SAFE_CALL(cudaMemcpy2DToArray(_arr, 0, 0, source, size.x * sizeof(T), size.x * sizeof(T), size.y, cudaMemcpyDeviceToDevice));
        has_data = true;
    }

    cudaTextureObject_t tex() const {
        if (!has_array) {
            throw std::runtime_error{"Texture array not set"};
        }
        if (!has_data) {
            throw std::runtime_error{"Texture data not set"};
        }
        return _tex;
    }

    ~BoundTexture2D() {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(_tex));
        if (has_array)
            CUDA_SAFE_CALL(cudaFreeArray(_arr));
    }

    BoundTexture2D& operator=(BoundTexture2D const&) = delete;
    BoundTexture2D(const BoundTexture2D& that) = delete;
};

template <typename T, int flag = cudaArrayDefault>
struct BoundTexture3D {
  private:
    cudaTextureDesc textureDescriptor;
    cudaArray_t _arr;
    cudaTextureObject_t _tex;
    bool has_array;
    bool has_data;

  public:
    const int3 size;
    BoundTexture3D(const int3 size,
                   const cudaTextureAddressMode addressMode,
                   const cudaTextureFilterMode filterMode,
                   const cudaTextureReadMode readMode = cudaReadModeElementType)
        : textureDescriptor{},
          size(size),
          has_array(false),
          has_data(false) {
        // Setup a texture descriptor
        textureDescriptor.addressMode[0] = addressMode;
        textureDescriptor.addressMode[1] = addressMode;
        if (flag == cudaArrayDefault)
            textureDescriptor.addressMode[2] = addressMode;
        textureDescriptor.filterMode = filterMode;
        textureDescriptor.readMode = readMode;
        textureDescriptor.normalizedCoords = 0;

        const cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

        CUDA_SAFE_CALL(cudaMalloc3DArray(&_arr, &channelDesc, extent, flag));

        // create texture object
        cudaResourceDesc resourceDescriptor = {};
        resourceDescriptor.resType = cudaResourceTypeArray;
        resourceDescriptor.res.array.array = _arr;

        // Create the texture object
        CUDA_SAFE_CALL(cudaCreateTextureObject(&_tex, &resourceDescriptor, &textureDescriptor, NULL));
        has_array = true;
    }

    void setData(const T* source) {
        if (!has_array)
            throw std::runtime_error("texture has no array");

        const cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);

        cudaMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, extent.width * sizeof(T),
                                                extent.width, extent.height);
        copyParams.dstArray = _arr;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;

        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        has_data = true;
    }

    void freeArray() {
        if (!has_array)
            return;

        CUDA_SAFE_CALL(cudaDestroyTextureObject(_tex));
        CUDA_SAFE_CALL(cudaFreeArray(_arr));
        has_array = false;
    }

    cudaTextureObject_t tex() const {
        if (!has_array) {
            throw std::runtime_error{"Texture array not set"};
        }
        if (!has_data) {
            throw std::runtime_error{"Texture data not set"};
        }
        return _tex;
    }

    ~BoundTexture3D() {
        freeArray();
    }

    BoundTexture3D& operator=(BoundTexture3D const&) = delete;
    BoundTexture3D(const BoundTexture3D& that) = delete;
};
