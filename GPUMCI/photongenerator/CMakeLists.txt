set(Headers)

set(Sources
	dummy.cpp)
	
set(CudaFiles
	PhotonGeneratorSimulated.cuh
	PhotonGeneratorSPECT.cuh
	PhotonGeneratorPhaseSpace.cuh
    PhotonGeneratorGainImage.cuh
	PhotonGeneratorUniform.cuh
    PhotonGeneratorUniformMorton.cuh)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

source_group ("CUDA Source Files" FILES ${CudaFiles})

CUDA_ADD_LIBRARY(photongenerator STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)
