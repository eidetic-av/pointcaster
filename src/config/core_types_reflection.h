#pragma once

#include <cstdint>
#include <pointcaster/core_types.h>
#include <rfl/internal/has_reflector.hpp>

// reflect-cpp specialisations for the core types

namespace rfl {

// pc::position needs a specialisation to ignore its padding value
template <> struct Reflector<pc::position> {
  struct ReflType {
    int16_t x = 0;
    int16_t y = 0;
    int16_t z = 0;
  };

  static pc::position to(const ReflType &v) noexcept {
    return pc::position{v.x, v.y, v.z, 0};
  }

  static ReflType from(const pc::position &p) { return {p.x, p.y, p.z}; }
};

// a pc::length is a distance, so it serialises as the bare millimetre count
// rather than as an object wrapping one
template <> struct Reflector<pc::length> {
  using ReflType = int16_t;

  static pc::length to(const ReflType &mm) noexcept { return pc::length{mm}; }

  static ReflType from(const pc::length &l) { return l.mm; }
};

// and a pc::radius is one too, in unsigned millimetres
template <> struct Reflector<pc::radius> {
  using ReflType = uint16_t;

  static pc::radius to(const ReflType &mm) noexcept { return pc::radius{mm}; }

  static ReflType from(const pc::radius &r) { return r.mm; }
};

} // namespace rfl
