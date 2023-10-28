#pragma once

#include <functional>
#include <serdepp/serde.hpp>
#include <tweeny/tweeny.h>
#include <unordered_map>

namespace pc::tween {

struct TweenConfiguration {
  int duration_ms = 300;
  tweeny::easing::enumerated ease_function =
      tweeny::easing::enumerated::quadraticOut;

  DERIVE_SERDE(TweenConfiguration,
               (&Self::duration_ms, "duration_ms")
	       (&Self::ease_function, "ease_function"))
};

} // namespace pc::tween
