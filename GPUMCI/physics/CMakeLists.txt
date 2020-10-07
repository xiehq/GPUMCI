set(Headers
	CudaSettings.h
    MaterialEntry.h)

set(Sources
	dummy.cpp)
	
set(CudaFiles
	CudaMonteCarlo.cuh)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

source_group ("CUDA Source Files" FILES ${CudaFiles})

CUDA_ADD_LIBRARY(physics STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)
