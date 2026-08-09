#include "zmq_context.h"

namespace pc::networking {

zmq::context_t &zmq_context() {
  constexpr int io_thread_count = 2;
  static zmq::context_t context{io_thread_count};
  return context;
}

} // namespace pc::networking