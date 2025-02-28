#pragma once

#include <memory>
#include <mutex>
#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>

namespace network_globals {

inline zmq::context_t &get_zmq_context() {
  static std::unique_ptr<zmq::context_t> ctx;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { ctx = std::make_unique<zmq::context_t>(); });
  return *ctx;
}

}; // namespace network_globals