set(Headers
	InteractionTypes.h
	PrecomputedInteractionUtils.h)

set(Sources
	dummy.cpp)
	
set(CudaFiles
    WoodcockStep.cuh
    FixedStep.cuh
	ComptonEverett.cuh
	ComptonPrecomputed.cuh
	InteractionHandlerPhoton.cuh
	InteractionHandlerAttenuating.cuh
    InteractionHandlerProjector.cuh
	NoInteraction.cuh
	PhotonPhoto.cuh
	RayleighPrecomputed.cuh)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

source_group ("CUDA Source Files" FILES ${CudaFiles})

CUDA_ADD_LIBRARY(interactions STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)
