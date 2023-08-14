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

cuda_add_library(interactions STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)
set_target_properties(interactions PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

