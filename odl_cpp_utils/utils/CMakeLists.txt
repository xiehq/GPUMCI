set(Headers
	cast.h
    EigenUtils.h
    Ellipse.h
    function_traits.h
    OptimizationUtils.h
    Phantom.h
    PrintArray.h
    Projection.h
    RLGeometries.h
    StandardPhantoms.h
    Statistics.h
    Timing.h
    utilities_export.h)

set(Sources
    dummy.cpp
    Phantom.cpp)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

CUDA_ADD_LIBRARY(odlcpputils_utils STATIC
	${Headers}
    ${Sources})

set_property(TARGET odlcpputils_utils PROPERTY FOLDER odl_cpp_utils)