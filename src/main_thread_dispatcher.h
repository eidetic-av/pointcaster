#pragma once

#include <concurrentqueue/concurrentqueue.h>
#include <functional>
#include <memory>

namespace pc {

class MainThreadDispatcher {
public:
  static void init() {
    _instance = std::make_unique<MainThreadDispatcher>();
  }

  static bool enqueue(const std::function<void()> &function) {
    return _instance->_dispatch_queue.enqueue(function);
  }

  static bool try_dequeue(std::function<void()> &function) {
    return _instance->_dispatch_queue.try_dequeue(function);
  }

private:
  inline static std::unique_ptr<MainThreadDispatcher> _instance;
  moodycamel::ConcurrentQueue<std::function<void()>> _dispatch_queue;
};

} // namespace pc
