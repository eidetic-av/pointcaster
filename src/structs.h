#pragma once

#include <pointclouds.h>
#include <serdepp/serde.hpp>

namespace pc::types {

template <typename T> struct MinMax {
  T min;
  T max;
  T *arr() { return &min; }
  bool contains(T value) const {
    if (value < min)
      return false;
    if (value > max)
      return false;
    return true;
  }
  DERIVE_SERDE(MinMax<T>,
	       (&Self::min, "min")(&Self::max, "max"))
};

template <typename T> struct VectorSize;

struct Float2 {
  using vector_type = float;
  vector_type x, y = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for Float3");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for Float3");
    }
  }
  bool operator==(const Float2 other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const Float2 other) const { return !operator==(other); }
  DERIVE_SERDE(Float2, (&Self::x, "x")(&Self::y, "y"))
};

template <> struct VectorSize<Float2> {
  static constexpr std::size_t value = 2;
};

struct Float3 {
  using vector_type = float;
  vector_type x, y, z = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for Float3");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for Float3");
    }
  }
  bool operator==(const Float3 other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator!=(const Float3 other) const { return !operator==(other); }
  DERIVE_SERDE(Float3,
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<Float3> {
  static constexpr std::size_t value = 3;
};

struct Float4 {
  using vector_type = float;
  vector_type w, x, y, z = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      case 3: return w;
      default: throw std::out_of_range("Index out of range for Float4");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      case 3: return w;
      default: throw std::out_of_range("Index out of range for Float4");
    }
  }
  bool operator==(const Float4 other) const {
    return x == other.x && y == other.y && z == other.z && w == other.w;
  }
  bool operator!=(const Float4 other) const { return !operator==(other); }
  DERIVE_SERDE(Float4,(&Self::w, "w")(&Self::x, "x")(
			   &Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<Float4> {
  static constexpr std::size_t value = 4;
};

struct Int2 {
  using vector_type = int;
  vector_type x, y = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for int2");
    }
  }
  const int& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for int2");
    }
  }
  bool operator==(const Int2 other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const Int2 other) const { return !operator==(other); }
  DERIVE_SERDE(Int2,(&Self::x, "x")(&Self::y, "y"))
};

template <> struct VectorSize<Int2> {
  static constexpr std::size_t value = 2;
};

struct Int3 {
  using vector_type = int;
  vector_type x, y, z = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for Int3");
    }
  }
  const int& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for Int3");
    }
  }
  bool operator==(const Int3 other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator!=(const Int3 other) const { return !operator==(other); }
  DERIVE_SERDE(Int3, 
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<Int3> {
  static constexpr std::size_t value = 3;
};

struct Uint2 {
  using vector_type = unsigned int;
  vector_type x, y = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for Uint2");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for Uint2");
    }
  }
  bool operator==(const Uint2 other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const Uint2 other) const { return !operator==(other); }
  DERIVE_SERDE(Uint2,(&Self::x, "x")(&Self::y, "y"))
};

template <> struct VectorSize<Uint2> {
  static constexpr std::size_t value = 2;
};

struct Short3 {
  using vector_type = short;
  vector_type x, y, z = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for Short3");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for Short3");
    }
  }
  bool operator==(const Short3 other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator!=(const Short3 other) const { return !operator==(other); }
  DERIVE_SERDE(Short3,
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<Short3> {
  static constexpr std::size_t value = 3;
};

#ifndef __CUDACC__

// concept to check if a type can be considered a vector type.
template <typename T>
concept IsVectorType = requires { typename T::vector_type; };

#endif

// just alias these for now before refactoring bob-pointclouds library
using PointCloud = bob::types::PointCloud;
using position = bob::types::position;
using color = bob::types::color;

// struct color {
//   using vector_type = unsigned char;
//   vector_type r, g, b, a = 0;
//   bool operator==(const color other) const {
//     return r == other.r && g == other.g && b == other.b && a == other.a;
//   }
//   bool operator!=(const color other) const { return !operator==(other); }
//   DERIVE_SERDE(color,
// 	       (&Self::r, "r")(&Self::g, "g")(&Self::b, "b")(&Self::a, "a"))
// };

} // namespace pc::types
