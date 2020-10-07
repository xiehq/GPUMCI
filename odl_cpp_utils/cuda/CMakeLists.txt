set(Headers
    cuda_utils.h
    cutil_math.h
    CudaMemory.h
    CuVector.h
    disableThrustWarnings.h
    enableThrustWarnings.h
    geometry.h
    texture.h
    thrustUtils.h)

set(Sources
	dummy.cpp)

set(CudaFiles
	CudaMemory.cu)
include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

CUDA_ADD_LIBRARY(odlcpputils_cuda STATIC
	${Headers}
    ${Sources}
    ${CudaFiles})

set_property(TARGET odlcpputils_cuda PROPERTY FOLDER odl_cpp_utils)
