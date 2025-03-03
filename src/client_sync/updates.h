#include "../structs.h"

#include <string>
#include <variant>

namespace pc::client_sync {

using array3f = std::array<float, 3>;
using pc::types::Float3;

enum class MessageType : uint8_t {
  Connected = 0x00,
  ClientHeartbeat = 0x01,
  ClientHeartbeatResponse = 0x02,
  ParameterUpdate = 0x10,
  ParameterRequest = 0x11
};

struct ParameterUpdate {
  std::string id;
  std::variant<float, int, Float3, array3f> value;
};

using SyncMessage = std::variant<MessageType, ParameterUpdate>;

} // namespace pc::client_sync