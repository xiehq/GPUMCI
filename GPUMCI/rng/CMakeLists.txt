set(Headers)

set(Sources
	dummy.cpp)
	
set(CudaFiles
    DeterministicRng.cuh
	CurandRng.cuh
	CurandRng.cu
	MWCRng.cuh)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

source_group ("CUDA Source Files" FILES ${CudaFiles})

cuda_add_library(rng STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)

set_target_properties(rng PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
