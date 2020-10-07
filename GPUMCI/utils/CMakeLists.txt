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

CUDA_ADD_LIBRARY(utils STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)
