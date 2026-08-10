#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <string>
#include <string_view>
#include <type_traits>
#include <util/string_map.h>
#include <variant>
#include <vector>

namespace pc::operators {

// the cloud types an operator can hand downstream alongside the main cloud.
using PipelineStream =
    std::variant<std::shared_ptr<PointCloud>, std::shared_ptr<VoxelisedCloud>,
                 std::shared_ptr<AabbList>>;

namespace detail {
template <typename T, typename Variant> struct is_stream_alternative;
template <typename T, typename... Alternatives>
struct is_stream_alternative<T, std::variant<Alternatives...>>
    : std::disjunction<std::is_same<std::shared_ptr<T>, Alternatives>...> {};
} // namespace detail

struct PipelineFrame {
  uint64_t seq = 0;
  std::shared_ptr<PointCloud> cloud;
  StringMap<PipelineStream> additional_streams;

  template <typename T>
  void set_stream(std::string label, std::shared_ptr<T> stream) {
    static_assert(detail::is_stream_alternative<T, PipelineStream>::value,
                  "stream type must be an alternative of PipelineStream");
    additional_streams.insert_or_assign(std::move(label), std::move(stream));
  }

  template <typename T = PointCloud>
  std::shared_ptr<T> stream(std::string_view label) const {
    static_assert(detail::is_stream_alternative<T, PipelineStream>::value,
                  "stream type must be an alternative of PipelineStream");
    const auto it = additional_streams.find(label);
    if (it == additional_streams.end()) return nullptr;
    const auto *held = std::get_if<std::shared_ptr<T>>(&it->second);
    return held ? *held : nullptr;
  }

  bool has_stream(std::string_view label) const {
    return additional_streams.contains(label);
  }

  // a mutable copy to modify and hand downstream
  std::shared_ptr<PipelineFrame> clone() const {
    return std::make_shared<PipelineFrame>(*this);
  }
};

using PipelineFramePtr = std::shared_ptr<const PipelineFrame>;

} // namespace pc::operators
