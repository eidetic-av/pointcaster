#include "uuid.h"

namespace pc::uuid {

const std::string generate() {
  uuid_t uuid;
  uuid_generate_random(uuid);
  char uuid_str[37]; // 36 characters plus null terminator
  uuid_unparse_lower(uuid, uuid_str);
  return std::string(uuid_str);
}

const std::string word(int32_t min_chars, int32_t max_chars, std::string prefix,
                       std::string suffix) {
  static wtf::generator generator;
  return generator.generateWord(min_chars, max_chars, prefix, suffix);
}

} // namespace pc::uuid
