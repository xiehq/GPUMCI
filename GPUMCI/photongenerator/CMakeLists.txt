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

add_library(photongenerator STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)

set_target_properties(photongenerator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)