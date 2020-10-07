GPUMCI
======

GPUMCI (Graphics Processing Unit Monte Carlo Imaging) is a code package for GPU based photon simulations, indended to be used in imaging applications such as CBCT.

Building
--------------------
The project uses CMake to enable builds. Using *cmake-gui* is recommended

[Cmake webpage](http://www.cmake.org/)

#### Unix:
Start by going the the directory where you want your binaries and run, 

    cmake-gui PATH_TO_SOURCE
    
And then set the required variables, to build, run
    
    make
    
To install the package to your python installation, run

    make pyinstall

#### Windows

To build on windows, open the CMake gui, run configure-generate and set the required variables. Then open the project with Visual Studio and build `PyInstall`.

#### ODL bindings

GPUMCI has [odl](https://github.com/odlgroup/odl) bindings available in gpumciodl. These can be installed by

    cd gpumciodl
    python setup.py install
    
Each of the odl style simulators has examples, runnable with for example

    python CudaProjectorPrimary.py

Running
--------------------
The current code has no Unit tests, but integration tests is performed using the bindings.

Code guidelines
--------------------
The code is written in C++11/14.

### Compilers
The code is intended to be usable with all major compilers. Current status (2015-06-22) is

| Platform     	| Compiler 	| Cuda 	| Compute 	| Works 	|
|--------------	|----------	|------	|---------	|-------	|
| Windows 7    	| VS2013   	| 7.0  	| 5.2     	| ✔     	|
| Windows 7    	| VS2013   	| 7.5  	| 5.2     	| ✔     	|
| Windows 10   	| VS2015   	| 7.0  	| 5.2     	| TODO  	|
| Fedora 21    	| GCC 4.9  	| 7.0  	| 5.2     	| ✔    |
| Ubuntu 14.04 	| GCC 4.9  	| 7.0  	| 5.2     	| ✔     	|
| Mac OSX 	    | ???      	| 7.0  	| 5.2     	| TODO    |

### Formating
The code is formatted using [CLang format](http://clang.llvm.org/docs/ClangFormat.html) as provided by the LLVM project. The particular style used is defined in the [formatting file](_clang-format).

External Dependences
--------------------
Current external dependencies are

#####Python
To build the Python bindings, GPUMCI needs acces to both python and numpy header files and compiled files to link against.

#####Eigen 
A C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. The code uses Eigen in conjunction with CUDA, which is only supported on the dev branch.

[Eigen webpage](http://eigen.tuxfamily.org)

#####Boost 
General library with C++ code. This project specifically uses [Boost.Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/index.html) to handle the python bindings.

[Boost webpage](http://www.boost.org/)
[Prebuilt boost (windows)](http://boost.teeks99.com/), this version uses python 2.7

#####CUDA
Used for GPU accelerated versions of algorithms. The code uses C++11 features in device code, so CUDA 7.0 is required. CUDA 6.5 may work on some platforms (notably windows). Since texture objects are used, a GPU of the Kepler architecture with compute capability 3.0 or higher is needed.

[CUDA](https://developer.nvidia.com/cuda-downloads)

Design
--------------------
The package is designed in a template based manner. The main code runner is in [CudaMonteCarlo.cuh](GPUMCI/physics/CudaMonteCarlo.cuh), this is however only a template. To compile this template, several structures have to be provided, a photon generator, a random number generator, a scorer (detector) and a interaction handler. These are discussed below.

#### PhotonGenerator

The photongenerator has to follow the following pattern

```cpp
struct PhotonGenerator {
  typedef IMPLEMENTATION_DEFINED SharedData; //CUDA shared data to share for example a photon buffer between threads
  
  __device__ void init(int idx, SharedData& sharedData, const CudaParameters& c_param){
    // Initialize the generator, for example load particle from global memory
  }
  
  template <typename Rng>
  __device__ bool generatePhoton(CudaMonteCarloParticle& photon, int idx, Rng& rng, SharedData& sharedData, const CudaParameters& c_param){
    // Writes the starting position, direction, energy and weight to photon. 
    // Returns true if a photon was generated, or false if no more photons will be generated.
  }
};
```

Several example implementations of this concept is given in [photongenerator](GPUMCI/photongenerator).

#### Rng

The rng should support generation of unsigned integers and floats, it has the following pattern

```cpp
struct Rng {
  //idx is the index of the current thread
  __device__ void init(int idx){
    // Initialize the rng, for example load a state from global memory
  }
  
  __device__ unsigned rand_int(){
    // Returns a random unsigned integer
  }
  
  __device__ float rand(){
    // Returns a random float in [0,1]
  }
  
  __device__ void saveState(int idx){
    // Saves the current state of the rng so that it may continue later
  }
};
```

#### Scorer

The scorer is used to score/save the photon once it leaves the volume.

```cpp
struct Scorer {
  __device__ void scoreDetector(const CudaMonteCarloParticle& photon, bool primary){
    //Score the given photon, primary indicates if it has scattered or not.
  }
}
```

#### InteractionHandler

The interactionHandler specifies how the photon should behave in interactions

```cpp

struct InteractionHandlerAttenuating {
    /**
	* Simulates an interaction
	*
	* Parameters:
	*		meanFreePathCM	The woodcock mean free path (in CM) at the energy of interaction
	*		myMedium          Index of the medium at the point of interaction
	*		myDensity			Density (g/cm^3) at the point of interaction
	*		primary				 Boolean (which will be written to) indicating if the photon is a primary (non-scattered photon). This should be written to if an interaction occurs.
	*   photon            The photon which should interact. The result is written inplace
	*		rng               The random number generator to use.
	*/
    template <typename Rng>
    __device__ void simulateInteraction(const float meanFreePathCM,
                                        const uint8_t myMedium,
                                        const float myDensity,
                                        bool& primary,
                                        CudaMonteCarloParticle& photon,
                                        Rng& rng,
                                        const CudaParameters& c_param) const {
      //Handle the interaction
    }
};
```

Troublefinding
--------------
There are a few common errors encountered, this is the solution to some of these

## Installation
* If, when compiling, you get a error like
    
        NumpyConfig.h not found
    
    then it is likely that the variable `PYTHON_NUMPY_INCLUDE_DIR` is not correctly set.

* If, when compiling, you get an error that begins with
    
        [ 20%] Building NVCC (Device) object RLcpp/CMakeFiles/PyCuda.dir//./PyCuda_generated_cuda.cu.o /usr/include/c++/4.9.2/bits/alloc_traits.h(248): error: expected a ">"
    
    It may be that you are trying to compile with CUDA 6.5 and GCC 4.9, this combination is not supported by CUDA.

* If you get a error like
    
        Error	5	error LNK2019: unresolved external symbol "__declspec(dllimport) struct _object * __cdecl boost::python::detail::init_module(struct PyModuleDef &,void (__cdecl*)(void))" (__imp_?init_module@detail@python@boost@@YAPEAU_object@@AEAUPyModuleDef@@P6AXXZ@Z) referenced in function PyInit_PyUtils	C:\Programming\Projects\RLcpp_bin\RLcpp\utils.obj	PyUtils
    
    then it is likely that you are trying to build against unmatched python header files and boost python version

## Running

* If, when running the tests in python, you get an error like
    
        RuntimeError: function_attributes(): after cudaFuncGetAttributes: invalid device function
        
    It may be that the compute version used is not supported by your setup, try changing the cmake variable `CUDA_COMPUTE`.

* If, when running the test in python, you encounter an error like

        ImportError: No module named GPUMCI
    
    It may be that you have not installed the package, run `make PyInstall` or equivalent.
