#include <atomic>
#include <memory>

namespace pc {

template <typename T> class ConcurrentContainer {
  std::atomic<std::shared_ptr<const T>> _value;

public:
  void store(std::shared_ptr<const T> cfg) {
    _value.store(std::move(cfg), std::memory_order_release);
  }
  std::shared_ptr<const T> load() const {
    return _value.load(std::memory_order_acquire);
  }
};

} // namespace pc