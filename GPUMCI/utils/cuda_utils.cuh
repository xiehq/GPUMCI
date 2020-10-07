#pragma once

namespace gpumci {
namespace cuda {
namespace util {

template <typename T>
inline __host__ __device__ void swap(T& f1, T& f2) {
    T temp = f1;
    f1 = f2;
    f2 = temp;
}

//------------------------------------------------------------------------------------------------
//
// Function : findVolumeIntersection
//
// Finds the intersection of the ray
//
//     start   +   t*direction
//
// with the volume.
//
// Code is from:
// Source: http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-7-intersecting-simple-shapes/ray-box-intersection/
//------------------------------------------------------------------------------------------------
inline __device__ bool findVolumeIntersection(const float3& start,
                                              const float3& direction,
                                              const float3& volumeMin,
                                              const float3& volumeMax,
                                              float3& intersection) {
    float tmin = (volumeMin.x - start.x) / direction.x;
    float tmax = (volumeMax.x - start.x) / direction.x;
    if (tmin > tmax)
        swap(tmin, tmax);

    float tymin = (volumeMin.y - start.y) / direction.y;
    float tymax = (volumeMax.y - start.y) / direction.y;
    if (tymin > tymax)
        swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (volumeMin.z - start.z) / direction.z;
    float tzmax = (volumeMax.z - start.z) / direction.z;

    if (tzmin > tzmax)
        swap(tzmin, tzmax);
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    if (tmax < 0)
        return false;

    float eps = 0.001; //Make sure the particle is really on the inside.
    intersection = start + (tmin + eps) * direction;

    return true;
}
}
}
}
