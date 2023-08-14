set(Headers
	)

set(Sources
	dummy.cpp)
	
set(CudaFiles
	DoseDetector.cuh
	NoDetector.cuh
	DetectorCBCT.cuh
	DetectorCBCTScatter.cuh
    DetectorStorePhoton.cuh)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

source_group ("CUDA Source Files" FILES ${CudaFiles})

cuda_add_library(detector STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)

set_target_properties(detector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
