#pragma once
#include <mutex>

namespace pc::profiling {

#ifdef TRACY_ENABLE
#  ifndef __CUDACC__
#    include <tracy/Tracy.hpp>

// when Tracy is enabled (and this isn’t a CUDA TU), use the real Tracy lockable
template <typename T>
using Lockable = tracy::Lockable<T>;

using mutex = Lockable<std::mutex>;

#    define PC_PROFILING_MUTEX(name) \
TracyLockableN(std::mutex, name, #name)

#  else   // __CUDACC__ (nvcc) — no Tracy C++ API, just plain types
template <typename T>
using Lockable = T;
using mutex = Lockable<std::mutex>;
#    define PC_PROFILING_MUTEX(name) std::mutex name
#  endif

#else  // !TRACY_ENABLE

// tracy disabled: everything collapses to the underlying type.
template <typename T>
using Lockable = T;

using mutex = Lockable<std::mutex>;
#  define PC_PROFILING_MUTEX(name) std::mutex name

#endif

} // namespace pc::profiling
