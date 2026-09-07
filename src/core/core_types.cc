#include <pointcaster/core_types.h>

#include <type_traits>
#include <zpp_bits.h>

namespace pc {

// all of this ensures that zpp bits can take fast-paths on de/serializing our
// point cloud structures

static_assert(std::is_trivially_copyable_v<position> &&
              std::has_unique_object_representations_v<position> &&
              sizeof(position) == 8);

static_assert(std::is_trivially_copyable_v<color> &&
              std::has_unique_object_representations_v<color> &&
              sizeof(color) == 4);

static_assert(std::is_trivially_copyable_v<scale> &&
              std::has_unique_object_representations_v<scale> &&
              sizeof(scale) == 2);

static_assert(zpp::bits::concepts::byte_serializable<position>);
static_assert(zpp::bits::concepts::byte_serializable<color>);
static_assert(zpp::bits::concepts::byte_serializable<scale>);

} // namespace pc