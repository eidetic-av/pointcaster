#pragma once
#include "../serialization.h"
#include "operator_types.h"

namespace pc::operators {

struct SampleFilterOperatorConfiguration {

  bool unfolded = false;
  bool enabled = true;
  int sample_count = 1;

  DERIVE_SERDE(SampleFilterOperatorConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::enabled, "enabled")
	       (&Self::sample_count, "sample_count")
	       )

  using MemberTypes = pc::reflect::type_list<bool, bool, int>;
  static const std::size_t MemberCount = 3;

  static constexpr const char* Name = "Sample Filter";
};

struct SampleFilterOperator : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  SampleFilterOperatorConfiguration _config;

  SampleFilterOperator(const SampleFilterOperatorConfiguration &config)
      : _config(config){};

  __device__ bool operator()(indexed_point_t point) const;

  static void draw_imgui_controls(SampleFilterOperatorConfiguration &config);
};

} // namespace pc::operators
