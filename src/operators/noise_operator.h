#pragma once
#include "../serialization.h"
#include "operator_types.h"

namespace pc::operators {

struct NoiseOperatorConfiguration {

  bool unfolded = false;
  bool enabled = true;
  float scale = 0.001f;
  float magnitude = 500.0f;
  int seed = 99;
  int repeat = 32;
  float lacunarity = 2.0f;
  float decay = 0.5f;

  DERIVE_SERDE(NoiseOperatorConfiguration,
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

  static constexpr const char* Name = "Noise";
};

struct NoiseOperator : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  NoiseOperatorConfiguration _config;

  NoiseOperator(const NoiseOperatorConfiguration &config)
      : _config(config){};

  __device__ indexed_point_t operator()(indexed_point_t point) const;

  static void draw_imgui_controls(NoiseOperatorConfiguration& config);
};

} // namespace pc::operators
