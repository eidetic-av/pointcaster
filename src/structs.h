#pragma once

#include <pointclouds.h>
#include "serialization.h"
#include <optional>

namespace pc::types {

template <typename T> struct VectorSize;

template<typename T>
struct MinMax {
  using vector_type = T;
  vector_type min, max;

  constexpr MinMax() = default;
  constexpr MinMax(T _min, T _max) : min(_min), max(_max){};

  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return min;
      case 1: return max;
      default: throw std::out_of_range("Index out of range for Float3");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return min;
      case 1: return max;
      default: throw std::out_of_range("Index out of range for Float3");
    }
  }

  bool operator==(const MinMax<T> other) const {
    return min == other.min && max == other.max;
  }
  bool operator!=(const MinMax<T> other) const { return !operator==(other); }
  
  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

  DERIVE_SERDE(MinMax<T>, (&Self::min, "min")(&Self::max, "max"))
};

template <typename T> struct VectorSize<MinMax<T>> {
  static constexpr std::size_t value = 2;
};

struct Float {
  using vector_type = float;
  vector_type value = 0;

  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return value;
      default: throw std::out_of_range("Index out of range for Float");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return value;
      default: throw std::out_of_range("Index out of range for Float");
    }
  }

  bool operator==(const Float other) const {
    return value == other.value;
  }
  bool operator!=(const Float other) const { return !operator==(other); }
  
  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

  DERIVE_SERDE(Float, (&Self::value, "value"))
};

template <> struct VectorSize<Float> {
  static constexpr std::size_t value = 1;
};

struct Float2 {
  using vector_type = float;
  vector_type x, y = 0;

  constexpr Float2() = default;
  constexpr Float2(vector_type _value) : x(_value), y(_value){};
  constexpr Float2(vector_type _x, vector_type _y) : x(_x), y(_y){};

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
  
  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

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

  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

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

  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

  DERIVE_SERDE(Float4,(&Self::w, "w")(&Self::x, "x")(
			   &Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<Float4> {
  static constexpr std::size_t value = 4;
};

struct Int {
  using vector_type = int;
  vector_type value = 0;

  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return value;
      default: throw std::out_of_range("Index out of range for Int");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return value;
      default: throw std::out_of_range("Index out of range for Int");
    }
  }

  bool operator==(const Int other) const {
    return value == other.value;
  }
  bool operator!=(const Int other) const { return !operator==(other); }
  
  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

  DERIVE_SERDE(Int, (&Self::value, "value"))
};

template <> struct VectorSize<Int> {
  static constexpr std::size_t value = 1;
};

struct Int2 {
  using vector_type = int;
  vector_type x, y = 0;

  constexpr Int2() = default;
  constexpr Int2(vector_type _value) : x(_value), y(_value){};
  constexpr Int2(vector_type _x, vector_type _y) : x(_x), y(_y){};

  constexpr Int2(std::initializer_list<vector_type> init) {
	  if (init.size() >= 2) {
	    auto iter = init.begin();
	    x = *iter;
	    y = *(++iter);
	  }
  }

  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for int2");
    }
  }
  const vector_type& operator[](std::size_t index) const {
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

  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

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
  const vector_type& operator[](std::size_t index) const {
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

  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

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

  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

  DERIVE_SERDE(Uint2,(&Self::x, "x")(&Self::y, "y"))
};

template <> struct VectorSize<Uint2> {
  static constexpr std::size_t value = 2;
};

struct Short2 {
  using vector_type = short;
  vector_type x, y = 0;

  constexpr Short2() = default;
  constexpr Short2(vector_type _value) : x(_value), y(_value){};
  constexpr Short2(vector_type _x, vector_type _y) : x(_x), y(_y){};

  constexpr Short2(std::initializer_list<vector_type> init) {
	  if (init.size() >= 2) {
	    auto iter = init.begin();
	    x = *iter;
	    y = *(++iter);
	  }
  }

  vector_type& operator[](std::size_t index) {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for Short2");
    }
  }
  const vector_type& operator[](std::size_t index) const {
    switch(index) {
      case 0: return x;
      case 1: return y;
      default: throw std::out_of_range("Index out of range for Short2");
    }
  }

  bool operator==(const Short2 other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const Short2 other) const { return !operator==(other); }

  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

  DERIVE_SERDE(Short2,(&Self::x, "x")(&Self::y, "y"))
};

template <> struct VectorSize<Short2> {
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

  vector_type* data() {
	  return reinterpret_cast<vector_type*>(this);
  }
  const vector_type* data() const {
	  return reinterpret_cast<const vector_type*>(this);
  }

  DERIVE_SERDE(Short3,
	       (&Self::x, "x")(&Self::y, "y")(&Self::z, "z"))
};

template <> struct VectorSize<Short3> {
  static constexpr std::size_t value = 3;
};

#ifndef __CUDACC__

// concept to check if a type can be considered a vector type.
template <typename T>
concept VectorType = requires { typename T::vector_type; };

template <typename T>
concept ScalarType = std::is_same_v<T, int> || std::is_same_v<T, short> ||
		       std::is_same_v<T, float> || std::is_same_v<T, double>;

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
