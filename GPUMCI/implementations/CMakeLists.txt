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

CUDA_ADD_LIBRARY(implementations STATIC 
	${Headers} 
	${Sources} 
	${CudaFiles}
)

target_link_libraries(implementations physics rng)
