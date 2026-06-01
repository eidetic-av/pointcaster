#pragma once

#include <rfl/DefaultVal.hpp>

namespace pc::networking {

struct PointStreamerConfiguration {
    rfl::DefaultVal<int> publish_hz = 240;
};

} // namespace pc::networking