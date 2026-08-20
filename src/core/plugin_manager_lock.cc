#include <pointcaster/plugin_manager_lock.h>

namespace pc {

std::recursive_mutex &plugin_manager_access() {
  static std::recursive_mutex access;
  return access;
}

} // namespace pc
