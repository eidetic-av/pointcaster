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

struct short3 {
  short x, y, z = 0;
  DERIVE_SERDE(short3,
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

struct float2 {
  float x, y = 0;

  float& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for float3");
    }
  }

  const float& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for float3");
    }
  }

  DERIVE_SERDE(float2, (&Self::x, "x")(&Self::y, "y"))
};

struct float3 {
  float x, y, z = 0;

  float& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for float3");
    }
  }

  const float& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default: throw std::out_of_range("Index out of range for float3");
    }
  }

  DERIVE_SERDE(float3,
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

struct float4 {
  float w, x, y, z = 0;
  DERIVE_SERDE(float4,(&Self::w, "w")(&Self::x, "x")(
			   &Self::y, "y")(&Self::z, "z"))
};

struct int3 {
  int x, y, z = 0;
  DERIVE_SERDE(int3, 
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

// just alias these for now before refactoring bob-pointclouds library
using PointCloud = bob::types::PointCloud;
using position = bob::types::position;


struct color {
  unsigned char r, g, b, a = 0;

  bool operator==(const color other) const {
    return r == other.r && g == other.g && b == other.b && a == other.a;
  }
  bool operator!=(const color other) const { return !operator==(other); }
  DERIVE_SERDE(color,
	       (&Self::r, "r")(&Self::g, "g")(&Self::b, "b")(&Self::a, "a"))
};

struct uint2 {
  unsigned int x, y = 0;

  unsigned int& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for uint2");
    }
  }

  const unsigned int& operator[](std::size_t index) const {
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

struct int2 {
  int x, y = 0;

  int& operator[](std::size_t index) {
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

} // namespace pc::types
