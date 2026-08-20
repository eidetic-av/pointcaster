#pragma once

#include <core/util/string_utils.h>
#include <string>

namespace pc {

// the prefix used when generating labels for a configuration type, taken from
// its reflection tag:
//  e.g. RangeFilterConfiguration "rangeFilter" becomes "range_filter"
template <typename ConfigType>
inline const std::string config_label_prefix =
    util::to_snake_case(typename ConfigType::Tag{}.name());

} // namespace pc
