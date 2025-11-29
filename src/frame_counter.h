#pragma once
#include <atomic>
#include <cstdint>

namespace pc::frame_counter {

inline std::atomic<std::uint64_t> g_frame_index{0};

inline std::uint64_t current_frame_index() noexcept {
  return g_frame_index.load(std::memory_order_relaxed);
}

inline std::uint64_t begin_new_frame() noexcept {
  return g_frame_index.fetch_add(1, std::memory_order_acq_rel) + 1;
}

} // namespace pc::frame_counter
