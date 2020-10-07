#pragma once

#include <Eigen/Core>
#include <ostream>
#include <iostream>
#include <algorithm> // std::minmax_element
#include <vector>

namespace SimRec2D {
namespace {
/**
	 * Picks colors uniformly, such that the minimum of the data is given the "lowest" color, and maximum the highest.
	 */
struct UniformColorPicker {
    UniformColorPicker(const std::string& characters, std::vector<long double> values)
        : _characters(characters) {
        auto result = std::minmax_element(values.begin(), values.end());
        _min = *result.first;
        _max = *result.second;

        _difference = _max - _min;
        _difference += std::numeric_limits<long double>::epsilon() * _difference * 2; //force the value into [0,1)

        if (_difference == 0.0)
            _difference = 1.0;
    }

    char pickColor(long double value) {
        long double val = (value - _min) / _difference;    //Calculate the normalized pixel value in [0,1)
        size_t index = (size_t)(val * _characters.size()); //Truncate the value to find a index in [0,1,2,...,numel(characters)-1]
        return _characters[index];
    }

    const std::string _characters;
    long double _min;
    long double _max;
    long double _difference;
};

/**
 * Picks color uniformly, but with compensation for outliers by using the interval between the 10th and 90th percentile.
 */
struct FancyColorPicker {
    FancyColorPicker(const std::string& characters, std::vector<long double> values)
        : _characters(characters) {
        std::vector<long double> valuesSorted(values);
        std::sort(valuesSorted.begin(), valuesSorted.end());

        _min = valuesSorted[(size_t)(valuesSorted.size() * 1.0 / characters.size())];
        _max = valuesSorted[(size_t)(valuesSorted.size() * (characters.size() - 1.0) / characters.size())];

        _difference = _max - _min;

        _min += _difference * 1.0 / characters.size();
        _max -= _difference * 1.0 / characters.size();

        if (_difference == 0.0)
            _difference = 1.0;
    }

    char pickColor(long double value) {
        if (value < _min)
            return _characters[0];
        if (value > _max)
            return _characters[_characters.size() - 1];
        else {
            long double val = (value - _min) / _difference;
            size_t index = 1 + (size_t)(val * (_characters.size() - 2));
            return _characters[index];
        }
    }

    const std::string _characters;
    long double _min;
    long double _max;
    long double _difference;
};
}

template <typename Derived>
Eigen::ArrayXXd shrink(const Eigen::DenseBase<Derived>& data, size_t width, size_t height) {
    // Calculate the image size and the stride
    double scaleX = (double)data.cols() / width;
    double scaleY = (double)data.rows() / height;

    ptrdiff_t windowX = std::max<ptrdiff_t>(1, static_cast<ptrdiff_t>(std::ceil(scaleX)));
    ptrdiff_t windowY = std::max<ptrdiff_t>(1, static_cast<ptrdiff_t>(std::ceil(scaleY)));

    Eigen::ArrayXXd result(height, width);
    for (ptrdiff_t i = 0; i < result.rows(); i++) {
        for (ptrdiff_t j = 0; j < result.cols(); j++) {
            result(i, j) = data.block(static_cast<size_t>(std::floor(i * scaleY)),
                                      static_cast<size_t>(std::floor(j * scaleX)),
                                      windowY, windowX).mean();
        }
    }

    return result;
}

/**
 * Prints a ASCII representation of the pixel values in @a image onto
 * @a stream.
 *
 * This function is primarily intended for debugging, testing, and
 * examples.  
 *
 * The width specifies the size of the output stream that should be used.
 * Restrict to stop the image from wrapping around the console. Uses sub-
 * sampling.
 *
 * @ingroup esp_image
 */
template <typename Derived>
inline void printArray(const Eigen::DenseBase<Derived>& dataIn,
                       std::ostream& stream = std::cout,
                       bool prettyPrint = true,
                       size_t width = 79,
                       size_t height = -1) {
    assert(dataIn.rows() > 0);
    assert(dataIn.cols() > 0);
    assert(dataIn.allFinite());
    assert(!dataIn.hasNaN());

    if (height == -1)
        height = width;

    Eigen::ArrayXXd data;
    if (prettyPrint) {
        if (dataIn.cols() == 1)
            data = shrink(dataIn.transpose(), std::min<size_t>(dataIn.rows(), width - 2), 1);
        else
            data = shrink(dataIn, std::min<size_t>(dataIn.cols(), width - 2), std::min<size_t>(dataIn.rows(), height - 2));
    } else
        data = shrink(dataIn, std::min<size_t>(dataIn.cols(), width), std::min<size_t>(dataIn.rows(), height));

    //Delimiters
    std::string top_line;
    std::string side_line;
    if (prettyPrint) {
        top_line.append("+");
        top_line.append(data.cols(), '-');
        top_line.append("+");
        top_line.append("\n");

        side_line = "|";
    }

    // Get patch min and max
    std::vector<long double> patchValues(data.data(), data.data() + data.size());

    //The characters we will use to represent the image
    const std::string characters(" .,-=oxOX8@#");
    FancyColorPicker colorPicker(characters, patchValues);

    //Print information
    if (prettyPrint) {
        auto minmaxel = std::minmax_element(patchValues.begin(), patchValues.end());

        stream << "Image: [" << dataIn.rows() << "x" << dataIn.cols() << "]"
               << " range: [" << *minmaxel.first << "," << *minmaxel.second << "]"
               << " Color scale: " << characters;

        stream << std::endl;
    }

    //Print an start delimiter
    stream << top_line;

    //Print the image
    for (ptrdiff_t i = 0; i < data.rows(); i++) {
        stream << side_line;
        for (ptrdiff_t j = 0; j < data.cols(); j++) {
            stream << colorPicker.pickColor(data(i, j));
        }
        stream << side_line;

        if (prettyPrint || i != data.rows() - 1)
            stream << std::endl;
    }

    //Print an end delimiter
    stream << top_line;
}
}
