#include <algorithm>
#include <core/logger/logger.h>
#include <pointcaster/task_pool.h>
#include <thread>

namespace pc {

BS::thread_pool<> &task_pool() {
  // constructed on first use, at the default size until preferences apply
  static BS::thread_pool<> pool{task_pool_default_size()};
  return pool;
}

std::size_t task_pool_default_size() {
  const auto hardware_threads = std::thread::hardware_concurrency() - 2;
  return hardware_threads > 0 ? hardware_threads : 1;
}

void set_task_pool_size(std::size_t thread_count) {
  const auto size = std::max<std::size_t>(thread_count, 1);
  if (task_pool().get_thread_count() == size) return;

  pc::logger()->debug("Resizing task pool from {} to {} threads",
                      task_pool().get_thread_count(), size);
  task_pool().reset(size);
}

} // namespace pc
