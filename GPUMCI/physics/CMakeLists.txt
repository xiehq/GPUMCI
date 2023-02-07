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

add_library(physics STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)

set_target_properties(physics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)