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

CUDA_ADD_LIBRARY(detector STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)
