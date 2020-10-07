#include <iostream>

#include <odl_cpp_utils/cuda/cutil_math.h>

inline std::ostream& operator<<(std::ostream& stream, const float2& vec) {
	return stream << vec.x << ", " << vec.y;
}

inline std::ostream& operator<<(std::ostream& stream, const float3& vec) {
	return stream << vec.x << ", " << vec.y << ", " << vec.z;
}

inline std::ostream& operator<<(std::ostream& stream, const int2& vec) {
	return stream << vec.x << ", " << vec.y;
}

inline std::ostream& operator<<(std::ostream& stream, const int3& vec) {
	return stream << vec.x << ", " << vec.y << ", " << vec.z;
}

#define CUDA_CHECK_ERRORS \
do {\
	cudaError err = cudaGetLastError(); \
if (cudaSuccess != err) {	\
	fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", \
	__FILE__, __LINE__, cudaGetErrorString(err)); \
	exit(-1); \
}\
} while (0)

#define gpuErrchk(ans) \
{ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
