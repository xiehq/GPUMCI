set(Headers
    numpy_eigen.h
    numpy_utils.h)

set(Sources
	dummy.cpp)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

CUDA_ADD_LIBRARY(odlcpputils_python STATIC
	${Headers}
    ${Sources})

set_property(TARGET odlcpputils_python PROPERTY FOLDER odl_cpp_utils)