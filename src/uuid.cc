#include "uuid.h"

namespace pc::uuid {

const std::string word(int32_t min_chars, int32_t max_chars, std::string prefix,
                       std::string suffix) {
  static wtf::generator generator;
  return generator.generateWord(min_chars, max_chars, prefix, suffix);
}

} // namespace pc::uuid
