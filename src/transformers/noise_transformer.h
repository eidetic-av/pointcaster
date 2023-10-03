#pragma once
#include "../serialization.h"
#include "transformer_types.h"

namespace pc::transformers {

struct NoiseTransformerConfiguration {

  bool unfolded = false;
  bool enabled = true;
  float scale = 0.001f;
  float magnitude = 500.0f;
  int seed = 99;
  int repeat = 32;
  float lacunarity = 2.0f;
  float decay = 0.5f;

  DERIVE_SERDE(NoiseTransformerConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::enabled, "enabled")
	       (&Self::scale, "scale")
	       (&Self::magnitude, "magnitude")
	       (&Self::seed, "seed")
	       (&Self::repeat, "repeat")
	       (&Self::lacunarity, "lacunarity")
	       (&Self::decay, "decay"))

  using MemberTypes = pc::reflect::type_list<bool, bool, float, float, int, int, float, float>;
  static const std::size_t MemberCount = 8;
};

struct NoiseTransformer : public thrust::unary_function<point_t, point_t> {

  NoiseTransformerConfiguration _config;

  NoiseTransformer(const NoiseTransformerConfiguration &config)
      : _config(config){};

  __device__ point_t operator()(point_t point) const;

  static void draw_imgui_controls(NoiseTransformerConfiguration& config);
};

} // namespace pc::transformers
