#pragma once

#include <cudautils/cutil_math.h>
#include <float.h>

namespace SimRec2D {
namespace cuda {
namespace detail {

__device__ inline bool intersects(const float2& boxMin,
                                  const float2& boxMax,
                                  const float2& x0,
								  const float2& direction,
                                  float& t0,
                                  float& t1) {
    t0 = -FLT_MAX;
    t1 = FLT_MAX;

    for (int i = 0; i < 2; i++) {
        float min_t;
        float max_t;

        switch (i) {
        case 0:
            min_t = (boxMin.x - x0.x) / direction.x;
            max_t = (boxMax.x - x0.x) / direction.x;
            break;
        case 1:
            min_t = (boxMin.y - x0.y) / direction.y;
            max_t = (boxMax.y - x0.y) / direction.y;
        }

        if (min_t > max_t) {
            t0 = fmaxf(max_t, t0);
            t1 = fminf(min_t, t1);
        } else {
            t0 = fmaxf(min_t, t0);
            t1 = fminf(max_t, t1);
        }

        if (t0 > t1)
            return false;
    }

    return true;
}

__device__ inline bool intersects(const float3& boxMin,
                                  const float3& boxMax,
                                  const float3& x0,
                                  const float3& direction,
                                  float& t0,
                                  float& t1) {

    t0 = -FLT_MAX;
    t1 = FLT_MAX;

    for (int i = 0; i < 3; i++) {
        float min_t;
        float max_t;

        switch (i) {
        case 0:
            min_t = (boxMin.x - x0.x) / direction.x;
            max_t = (boxMax.x - x0.x) / direction.x;
            break;
        case 1:
            min_t = (boxMin.y - x0.y) / direction.y;
            max_t = (boxMax.y - x0.y) / direction.y;
            break;
        case 2:
            min_t = (boxMin.z - x0.z) / direction.z;
            max_t = (boxMax.z - x0.z) / direction.z;
        }

        if (min_t > max_t) {
            t0 = fmaxf(max_t, t0);
            t1 = fminf(min_t, t1);
        } else {
            t0 = fmaxf(min_t, t0);
            t1 = fminf(max_t, t1);
        }

        if (t0 > t1)
            return false;
    }

    return true;
}

} //detail
} //cuda
} //SimRec2D