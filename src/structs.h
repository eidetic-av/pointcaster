#pragma once

#include <pointclouds.h>
#include <serdepp/serde.hpp>

namespace pc::types {

template <typename T> struct minMax {
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
  DERIVE_SERDE(minMax<T>,
	       (&Self::min, "min")(&Self::max, "max"))
};

template <typename T> struct VectorSize;

struct float2 {
  using vector_type = float;
  vector_type x, y = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for float3");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for float3");
    }
  }
  bool operator==(const float2 other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const float2 other) const { return !operator==(other); }
  DERIVE_SERDE(float2, (&Self::x, "x")(&Self::y, "y"))
};

template <> struct VectorSize<float2> {
  static constexpr std::size_t value = 2;
};

struct float3 {
  using vector_type = float;
  vector_type x, y, z = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for float3");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for float3");
    }
  }
  bool operator==(const float3 other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator!=(const float3 other) const { return !operator==(other); }
  DERIVE_SERDE(float3,
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<float3> {
  static constexpr std::size_t value = 3;
};

struct float4 {
  using vector_type = float;
  vector_type w, x, y, z = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      case 3: return w;
      default: throw std::out_of_range("Index out of range for float4");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      case 3: return w;
      default: throw std::out_of_range("Index out of range for float4");
    }
  }
  bool operator==(const float4 other) const {
    return x == other.x && y == other.y && z == other.z && w == other.w;
  }
  bool operator!=(const float4 other) const { return !operator==(other); }
  DERIVE_SERDE(float4,(&Self::w, "w")(&Self::x, "x")(
			   &Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<float4> {
  static constexpr std::size_t value = 4;
};

struct int2 {
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
  bool operator==(const int2 other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const int2 other) const { return !operator==(other); }
  DERIVE_SERDE(int2,(&Self::x, "x")(&Self::y, "y"))
};

template <> struct VectorSize<int2> {
  static constexpr std::size_t value = 2;
};

struct int3 {
  using vector_type = int;
  vector_type x, y, z = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for int3");
    }
  }
  const int& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for int3");
    }
  }
  bool operator==(const int3 other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator!=(const int3 other) const { return !operator==(other); }
  DERIVE_SERDE(int3, 
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<int3> {
  static constexpr std::size_t value = 3;
};

struct uint2 {
  using vector_type = unsigned int;
  vector_type x, y = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for uint2");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for uint2");
    }
  }
  bool operator==(const uint2 other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const uint2 other) const { return !operator==(other); }
  DERIVE_SERDE(uint2,(&Self::x, "x")(&Self::y, "y"))
};

template <> struct VectorSize<uint2> {
  static constexpr std::size_t value = 2;
};

struct short3 {
  using vector_type = short;
  vector_type x, y, z = 0;
  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for short3");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for short3");
    }
  }
  bool operator==(const short3 other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator!=(const short3 other) const { return !operator==(other); }
  DERIVE_SERDE(short3,
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<short3> {
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
