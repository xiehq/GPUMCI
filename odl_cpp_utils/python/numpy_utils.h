#pragma once

// Disable deprecated API
#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numeric.hpp>
#include <exception>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <typeinfo>
#include <numeric>

using namespace boost::python;

template <typename T>
inline int getEnum();

#define makeDefinition(TYPE, NAME) \
    template <>                    \
    inline int getEnum<TYPE>() {   \
        return NAME;               \
    }
makeDefinition(float, NPY_FLOAT);
makeDefinition(double, NPY_DOUBLE);
makeDefinition(long double, NPY_LONGDOUBLE);
makeDefinition(bool, NPY_BOOL);
makeDefinition(char, NPY_BYTE);
makeDefinition(signed char, NPY_BYTE);
makeDefinition(unsigned char, NPY_UBYTE);
makeDefinition(signed short, NPY_SHORT);
makeDefinition(unsigned short, NPY_USHORT);
makeDefinition(signed int, NPY_INT);
makeDefinition(unsigned int, NPY_UINT);
makeDefinition(signed long, NPY_LONG);
makeDefinition(unsigned long, NPY_ULONG);
makeDefinition(signed long long, NPY_LONGLONG);
makeDefinition(unsigned long long, NPY_ULONGLONG);
#undef makeDefinition

template <typename T>
bool isTypeCompatible(const numeric::array& data) {
    PyArrayObject* a = (PyArrayObject*)data.ptr();

    if (a == NULL) return false;

    int data_array_type = PyArray_TYPE(a);
    int T_array_type = getEnum<T>();

    if (data_array_type == T_array_type)
        return true;
    else if (data_array_type == NPY_LONG && T_array_type == NPY_INT &&
             sizeof(int) == sizeof(long)) // We handle the case with long being
                                          // equal to int on windows
        return true;
    else if (data_array_type == NPY_INT && T_array_type == NPY_LONG &&
             sizeof(int) == sizeof(long))
        return true;
    else if (data_array_type == NPY_ULONG && T_array_type == NPY_UINT &&
             sizeof(unsigned int) == sizeof(unsigned long)) // We handle the case with long being
                                                            // equal to int on windows
        return true;
    else if (data_array_type == NPY_UINT && T_array_type == NPY_ULONG &&
             sizeof(unsigned int) == sizeof(unsigned long))
        return true;
    else
        return false;
}

struct EigenSize {
    size_t dataRows, dataCols, dimension, datadimension;
};

inline EigenSize getSize(const numeric::array& data) {
    const tuple shape = extract<tuple>(data.attr("shape"));
    const size_t datadimension = len(shape);
    size_t dimension = datadimension;

    size_t dataRows, dataCols;
    if (datadimension == 1) {
        dataRows = extract<int>(shape[0]);
        dataCols = 1;
    } else if (datadimension == 2) {
        dataRows = extract<int>(shape[0]);
        dataCols = extract<int>(shape[1]);

        if (dataRows == 1 || dataCols == 1) dimension = 1;
    } else
        throw std::invalid_argument("Dimension is not 1 or 2");

    EigenSize result = {dataRows, dataCols, dimension, datadimension};
    return result;
}

inline EigenSize getSizeGeneral(const object& data) {
    size_t dataRows = 1;
    size_t dataCols = 1;
    size_t dimension = 1;
    size_t dataDimension = 1;

    try {
        dataRows = len(data);
        if (PyObject_HasAttrString(object(data[0]).ptr(), "len")) {
            dataCols = len(data[0]);
            dataDimension = 2;

            // TODO check that all others are equal
        }
    } catch (const error_already_set&) {
        throw std::invalid_argument("Data is not of array type");
    }

    if (dataRows > 1 && dataCols > 1) dimension = 2;

    EigenSize result = {dataRows, dataCols, dimension, dataDimension};
    return result;
}

template <typename T, int N>
boost::python::numeric::array makeArray(npy_intp dims[N]) {
    PyObject* pyObj = PyArray_SimpleNew(N, dims, getEnum<T>());

    boost::python::handle<> handle(pyObj);
    boost::python::numeric::array arr(handle);

    return extract<boost::python::numeric::array>(arr.copy());
}

#ifdef ODL_MSVC_2012
template <typename T>
boost::python::numeric::array makeArray(npy_intp size) {
    npy_intp dims[1] = {size};
    return makeArray<T, 1>(dims);
}
template <typename T>
boost::python::numeric::array makeArray(npy_intp size_1, npy_intp size_2) {
    npy_intp dims[2] = {size_1, size_2};
    return makeArray<T, 2>(dims);
}
template <typename T>
boost::python::numeric::array makeArray(npy_intp size_1, npy_intp size_2,
                                        npy_intp size_3) {
    npy_intp dims[3] = {size_1, size_2, size_3};
    return makeArray<T, 3>(dims);
}
#else
template <typename T, typename... Sizes>
boost::python::numeric::array makeArray(Sizes... sizes) {
    npy_intp dims[sizeof...(sizes)] = {sizes...};
    return makeArray<T, sizeof...(sizes)>(dims);
}
#endif

template <typename T>
T* getDataPtr(const numeric::array& data) {
    PyArrayObject* a = (PyArrayObject*)data.ptr();

    if (a == NULL) throw std::invalid_argument("Could not get NP array.");

    // Check that type is correct
    if (!isTypeCompatible<T>(data))
        throw std::invalid_argument(("Expected element type " +
                                     std::string(typeid(T).name()) + ", got " +
                                     PyArray_DESCR(a)->type)
                                        .c_str());

    T* p = (T*)PyArray_DATA(a);

    return p;
}
