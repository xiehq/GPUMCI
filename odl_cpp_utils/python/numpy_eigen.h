#pragma once

#include "boost/python/numeric.hpp"
#include <numpy/arrayobject.h>
#include <Python.h>

#include <Eigen/Core>

#include <stdexcept>
#include <string>
#include <typeinfo>
#include <numeric>
#include <stdint.h>

#include <odl_cpp_utils/python/numpy_utils.h>

using namespace boost::python;

template <typename EigenArray>
void verifySize(const EigenSize& size) {
    static const int rowsAtCompile = EigenArray::RowsAtCompileTime;
    static const int colsAtCompile = EigenArray::ColsAtCompileTime;

    if (size.dimension == 1 &&
        ((rowsAtCompile != Eigen::Dynamic && rowsAtCompile != 1) &&
         (colsAtCompile != Eigen::Dynamic && colsAtCompile != 1))) {
        throw std::invalid_argument(("Dimensions not equal expected 2, got " + std::to_string(size.dataCols) + "x" + std::to_string(size.dataRows)).c_str());
    }
    if (size.dimension == 2 &&
        ((rowsAtCompile == 1) ||
         (colsAtCompile == 1))) {

        throw std::invalid_argument(("Dimensions not equal expected 1, got " + std::to_string(size.dataCols) + "x" + std::to_string(size.dataRows)).c_str());
    }

    if ((rowsAtCompile != Eigen::Dynamic && rowsAtCompile != size.dataRows) ||
        (colsAtCompile != Eigen::Dynamic && colsAtCompile != size.dataCols)) {
        throw std::invalid_argument(("Sizes not equal, expected " + std::to_string(rowsAtCompile) + "x" + std::to_string(colsAtCompile) + " got " + std::to_string(size.dataRows) + "x" + std::to_string(size.dataCols)).c_str());
    }
}

template <typename EigenArray>
bool isPtrCompatible(const numeric::array& numpyArray) {
    typedef typename EigenArray::Scalar Scalar;

    if (!isTypeCompatible<Scalar>(numpyArray)) {
        return false;
    }
    int flags = PyArray_FLAGS((PyArrayObject*)numpyArray.ptr());
    if (!(flags & NPY_ARRAY_ALIGNED)) {
        return false;
    }

    //Verify order
    if ((flags & NPY_ARRAY_F_CONTIGUOUS) && !EigenArray::IsRowMajor) {
        return true;
    } else if ((flags & NPY_ARRAY_C_CONTIGUOUS) && EigenArray::IsRowMajor) {
        return true;
    } else {
        return false;
    }
}

template <typename EigenArray>
Eigen::Map<EigenArray> mapInput(numeric::array data) {
    typedef typename EigenArray::Scalar Scalar;

    EigenSize size = getSize(data);
    verifySize<EigenArray>(size);
    if (isPtrCompatible<EigenArray>(data)) {
        return Eigen::Map<EigenArray>(getDataPtr<Scalar>(data), size.dataRows, size.dataCols);
    } else {
        throw std::invalid_argument(std::string("Array is not pointer compatible with Eigen Array"));
    }
}

template <typename EigenArray>
void copyElements(const EigenSize& size, const object& in, EigenArray& out) {
    typedef typename EigenArray::Scalar Scalar;

    if (size.datadimension == 1) {
        for (size_t i = 0; i < size.dataRows; i++)
            out(i) = extract<Scalar>(in[i]);
    } else {
        for (size_t i = 0; i < size.dataRows; i++) {
            for (size_t j = 0; j < size.dataCols; j++)
                out(i, j) = extract<Scalar>(in[i][j]);
        }
    }
}

template <typename InputType, typename EigenArray>
bool copy_if_valid(EigenArray& out, const numeric::array& dataArray) {
    typedef typename EigenArray::Scalar Scalar;
    if (isPtrCompatible<InputType>(dataArray)) {
        auto mapped = mapInput<InputType>(dataArray);
        out = mapped.template cast<Scalar>(); //If out type does not equal in type, perform a cast. If equal this is assignment.
        return true;
    } else
        return false;
}

template <typename EigenArray>
EigenArray copyInput(const object& data) {
    static const int rowsAtCompile = EigenArray::RowsAtCompileTime;
    static const int colsAtCompile = EigenArray::ColsAtCompileTime;
    static const bool staticSize = (rowsAtCompile != Eigen::Dynamic) && (colsAtCompile != Eigen::Dynamic);
    typedef typename EigenArray::Scalar Scalar;
    typedef typename Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorDoubleArray;
    typedef typename Eigen::Array<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorLongArray;
    typedef typename Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMajorDoubleArray;
    typedef typename Eigen::Array<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMajorLongArray;

    extract<numeric::array> asNumeric(data);
    if (asNumeric.check()) {
        //If data is an array, attempt some efficient methods first

        numeric::array dataArray = asNumeric();

        EigenSize size = getSize(dataArray);
        verifySize<EigenArray>(size);

        EigenArray out;
        if (!staticSize)
            out.resize(size.dataRows, size.dataCols);

        if (isPtrCompatible<EigenArray>(dataArray)) {
            //Use raw buffers if possible
            auto mapped = mapInput<EigenArray>(dataArray);
            out = mapped;
        } else if (isPtrCompatible<RowMajorDoubleArray>(dataArray)) {
            //Default implementation for double numpy array
            auto mapped = mapInput<RowMajorDoubleArray>(dataArray);
            out = mapped.cast<Scalar>(); //If out type does not equal in type, perform a cast. If equal this is assignment.
        } else if (isPtrCompatible<RowMajorLongArray>(dataArray)) {
            //Default implementation for long numpy array
            auto mapped = mapInput<RowMajorLongArray>(dataArray);
            out = mapped.cast<Scalar>(); //If out type does not equal in type, perform a cast. If equal this is assignment.
        } else if (isPtrCompatible<ColMajorDoubleArray>(dataArray)) {
            auto mapped = mapInput<ColMajorDoubleArray>(dataArray);
            out = mapped.cast<Scalar>(); //If out type does not equal in type, perform a cast. If equal this is assignment.
        } else if (isPtrCompatible<ColMajorLongArray>(dataArray)) {
            auto mapped = mapInput<ColMajorLongArray>(dataArray);
            out = mapped.cast<Scalar>(); //If out type does not equal in type, perform a cast. If equal this is assignment.
        } else {
            //Slow method if raw buffers unavailable.
            copyElements(size, data, out);
        }

        return out;
    } else {
        EigenSize size = getSizeGeneral(data);
        verifySize<EigenArray>(size);

        EigenArray out;
        if (!staticSize)
            out.resize(size.dataRows, size.dataCols);

        copyElements(size, data, out);

        return out;
    }
}

template <typename EigenArray>
numeric::array copyOutput(const EigenArray& data) {
    typedef typename EigenArray::Scalar Scalar;

    npy_intp dims[2] = {(npy_intp)data.rows(), (npy_intp)data.cols()};

    object obj(handle<>(PyArray_SimpleNew(2, dims, getEnum<Scalar>())));
    numeric::array arr = extract<numeric::array>(obj);

    auto mapped = mapInput<Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(arr); //Numpy uses Row Major storage
    mapped = data;

    return extract<numeric::array>(arr.copy()); //Copy to pass ownership
}

template <typename EigenType>
struct eigenarray_from_python_object {
    eigenarray_from_python_object() {
        converter::registry::push_back(&convertible, &construct, type_id<EigenType>());
    }

    static void* convertible(PyObject* obj_ptr) {
        auto ext_arr = extract<boost::python::numeric::array>(obj_ptr);
        auto ext_list = extract<boost::python::list>(obj_ptr);
        auto ext_tuple = extract<boost::python::tuple>(obj_ptr);
        if (ext_arr.check() || ext_list.check() || ext_tuple.check())
            return obj_ptr;
        else
            return 0;
    }

    static void construct(PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((converter::rvalue_from_python_storage<EigenType>*)data)->storage.bytes;
        object obj = extract<object>(obj_ptr);
        new (storage) EigenType(copyInput<EigenType>(obj));
        data->convertible = storage;
    }
};

template <typename EigenType>
struct eigenarray_to_python_object {

    eigenarray_to_python_object() {
        to_python_converter<EigenType,
                            eigenarray_to_python_object<EigenType>>();
    }

    static PyObject* convert(const EigenType& v) {
        return incref(copyOutput(v).ptr());
    }
};

template <typename EigenType>
void create_eigen_converter() {
    // check if inner type is in registry already
    const boost::python::type_info inner_info = type_id<EigenType>();
    const converter::registration* inner_registration = converter::registry::query(inner_info);
    if (inner_registration == 0 || inner_registration->m_to_python == 0) {
        // not already in registry
        eigenarray_to_python_object<EigenType>();
        eigenarray_from_python_object<EigenType>();
    } else {
        // already in registry
    }
}

template <typename Scalar, int rows, int cols>
void instantiate_eigen_conv() {
    create_eigen_converter<Eigen::Array<Scalar, rows, cols>>();
    create_eigen_converter<Eigen::Matrix<Scalar, rows, cols>>();
}

template <typename Scalar, int rows>
void instantiate_eigen_conv() {
    instantiate_eigen_conv<Scalar, rows, Eigen::Dynamic>();
    instantiate_eigen_conv<Scalar, rows, 1>();
    instantiate_eigen_conv<Scalar, rows, 2>();
    instantiate_eigen_conv<Scalar, rows, 3>();
    instantiate_eigen_conv<Scalar, rows, 4>();
}

template <typename Scalar>
void instantiate_eigen_conv() {
    instantiate_eigen_conv<Scalar, Eigen::Dynamic>();
    instantiate_eigen_conv<Scalar, 1>();
    instantiate_eigen_conv<Scalar, 2>();
    instantiate_eigen_conv<Scalar, 3>();
    instantiate_eigen_conv<Scalar, 4>();
}

//Create converters from python to C++ for eigen.
void export_eigen_conv() {
    instantiate_eigen_conv<double>();
    instantiate_eigen_conv<float>();
    instantiate_eigen_conv<int32_t>();
    instantiate_eigen_conv<int64_t>();
}
