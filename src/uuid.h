#include <string>
#include <uuid/uuid.h>

namespace pc::uuid {

const std::string generate() {
  uuid_t uuid;
  uuid_generate_random(uuid);
  char uuid_str[37]; // 36 characters plus null terminator
  uuid_unparse_lower(uuid, uuid_str);
  return std::string(uuid_str);
}

} // namespace pc::uuid
