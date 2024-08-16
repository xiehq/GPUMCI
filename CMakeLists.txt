# The name of our project is "HELLO". CMakeLists files in this project can 
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and 
# to the root binary directory of the project as ${HELLO_BINARY_DIR}. 
cmake_minimum_required (VERSION 2.6) 
project (GPUMCI)

#Set binary dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# add cmake find modules
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) 

# Use folders in MSVC
set_property(GLOBAL PROPERTY USE_FOLDERS true)

# Export compile for YCM
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add test project
enable_testing ()
add_definitions(-DCTEST_OUTPUT_ON_FAILURE)

# Add eigen
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_DONT_ALIGN_STATICALLY) #Causes lots of issues sadly and we dont use eigen for numerics.

# Add cuda
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(CUDA_COMPUTE "61" CACHE STRING "Cuda compute capability to compile for")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_${CUDA_COMPUTE},code=sm_${CUDA_COMPUTE})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lineinfo)

#these are for eigen should be set to target build but this is easier
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)

# Enable fast math
option(CUDA_USE_FAST_MATH "Enable fast math in cuda (can decrease precision)" TRUE)
if(CUDA_USE_FAST_MATH)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math)
endif()

# Add openMP
option(OPENMP_ENABLED "Enable OpenMP" TRUE)
if(OPENMP_ENABLED)
    find_package(OpenMP QUIET REQUIRED)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# C++ flags
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /bigobj")
    
    #Set warning level 3
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    endif()
else()
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
    CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++14 -Xcompiler -fPIC)
    elseif(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -Xcompiler -fPIC)
    elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++0x -Xcompiler -fPIC)
    else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++10 support. Please use a different C++ compiler.")
    endif()

    #Position independent code is needed to link shared-static
    set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)    

    #Set proper flag propagation for cuda
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)

    
endif()


# Python bindings
option(PYTHON_ENABLED "Create python bindings" FALSE)
if(PYTHON_ENABLED)
    set(Boost_USE_SHARED_LIBS   ON)
    set(Boost_USE_MULTITHREADED ON)
    find_package(Boost COMPONENTS
                python
                REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    
    # Find python
    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    
    # Find numpy
    find_package(Numpy REQUIRED)
    include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
    
    link_directories(${PYTHON_LIBRARIES})

    #disable old
    add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
endif(PYTHON_ENABLED)

# Dependencies
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/odl_cpp_utils/CMakeLists.txt")
    message(SEND_ERROR "odl_cpp_utils not initialized. Run 'git submodule update --init'." )
endif()
add_subdirectory(odl_cpp_utils)

# Add sub directories as needed
add_subdirectory(GPUMCI)
if (PYTHON_ENABLED)
    add_subdirectory(GPUMCIPy)
endif()

