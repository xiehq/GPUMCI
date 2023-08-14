set(Headers
	AbsorbingMC.h
	DoseMC.h
    GainMC.h
    GainDoseMC.h
    ForwardProjector.h
	SimpleMC.h
    PrecomputedMC.h
    PhaseSpaceMC.h
    SpectMC.h
    PhaseSpaceStorePhotonsMC.h)

set(Sources
	dummy.cpp)
	
set(CudaFiles
	WoodcockUtils.cuh
	WoodcockUtils.cu
	MaterialUtils.cuh
	MaterialUtils.cu
	AbsorbingMC.cu
	DoseMC.cu
    ForwardProjector.cu
    GainMC.cu
    GainDoseMC.cu
	SimpleMC.cu
	PrecomputedMC.cu
    PhaseSpaceMC.cu
    SpectMC.cu
    PhaseSpaceStorePhotonsMC.cu)


include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

source_group ("CUDA Source Files" FILES ${CudaFiles})

cuda_add_library(implementations STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)
set_target_properties(implementations PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(implementations PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(implementations physics rng)
