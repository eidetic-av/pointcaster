#include "uuid.h"
#include <string>
#include <random>

namespace pc::uuid {

const std::string word(int32_t min_chars, int32_t max_chars, std::string prefix,
                       std::string suffix) {
  static wtf::generator generator;
  return generator.generateWord(min_chars, max_chars, prefix, suffix);
}

unsigned long int digit() {
  static std::random_device rd;
  static std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned long int> distr;
  unsigned long int random_value = distr(eng);
  return random_value;
}

} // namespace pc::uuid
