#pragma once

#include <cstdint>
#include <cstdlib>
#include <tracy/Tracy.hpp>

namespace pc::profiling {

class MemoryTracked {
public:
  void *operator new(std::size_t count) {
    auto ptr = malloc(count);
    TracyAlloc(ptr, count);
    return ptr;
  }
  void operator delete(void *ptr) noexcept {
    TracyFree(ptr);
    free(ptr);
  }
};

} // namespace pc::profiling
