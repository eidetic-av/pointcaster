#pragma once

#include <zmq.hpp>

namespace pc::networking {

zmq::context_t &zmq_context();

}