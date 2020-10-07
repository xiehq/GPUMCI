#pragma once

#include <limits>
#include <stdexcept>
#include <type_traits>

template <typename I, typename J>
typename std::enable_if<std::is_signed<I>::value && std::is_signed<J>::value, I>::type
narrow_cast(const J& value) {
    if (value < std::numeric_limits<I>::min() || value > std::numeric_limits<I>::max()) {
        throw std::out_of_range("out of range");
    }
    return static_cast<I>(value);
}

template <typename I, typename J>
typename std::enable_if<std::is_signed<I>::value && std::is_unsigned<J>::value, I>::type
narrow_cast(const J& value) {
    if (value > static_cast<typename std::make_unsigned<I>::type>(std::numeric_limits<I>::max())) {
        throw std::out_of_range("out of range");
    }
    return static_cast<I>(value);
}

template <typename I, typename J>
typename std::enable_if<std::is_unsigned<I>::value && std::is_signed<J>::value, I>::type
narrow_cast(const J& value) {
    if (value < 0 || static_cast<typename std::make_unsigned<J>::type>(value) > std::numeric_limits<I>::max()) {
        throw std::out_of_range("out of range");
    }
    return static_cast<I>(value);
}

template <typename I, typename J>
typename std::enable_if<std::is_unsigned<I>::value && std::is_unsigned<J>::value, I>::type
narrow_cast(const J& value) {
    if (value > std::numeric_limits<I>::max()) {
        throw std::out_of_range("out of range");
    }
    return static_cast<I>(value);
}
