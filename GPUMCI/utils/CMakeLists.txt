set(Headers
    CudaCosWeighting.h
	physical_constants.h)

set(Sources
    CudaCosWeighting.cpp
	dummy.cpp)
	
set(CudaFiles
    CudaCosWeighting.cu
	cuda_utils.cuh)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

source_group ("CUDA Source Files" FILES ${CudaFiles})

cuda_add_library(utils STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)

set_target_properties(utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
