#pragma once

#include <mutex>
#include <pointcaster/core.h>

namespace pc {

// TODO idk revisit the need for a recursive_mutex...
// seems like some ownership or lifetimes are ripe for questioning
// if we are making use of this kind of thing...
POINTCASTER_CORE_EXPORT std::recursive_mutex &plugin_manager_access();

} // namespace pc
