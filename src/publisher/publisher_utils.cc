#include "publisher_utils.h"
#include <initializer_list>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace pc::publisher {

std::string
construct_topic_string(const std::string_view topic_name,
                       std::initializer_list<std::string_view> topic_nodes) {

  // a map to cache pre-constructed topic strings avoiding reconstruction
  static std::unordered_map<std::size_t, std::string> cache;
  static std::mutex cache_mutex;

  // compute the hash of the arguments
  std::hash<std::string_view> hasher;
  std::size_t hash = hasher(topic_name);
  for (const auto &node : topic_nodes) {
    // mix hash bits with each topic node
    hash ^= hasher(node) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }

  // check and return if our result is already cached
  {
    std::lock_guard lock(cache_mutex);
    auto it = cache.find(hash);
    if (it != cache.end()) {
      return it->second;
    }
  }

  // if it's not cached we construct the string

  // allocate the memory
  std::size_t result_string_length = topic_name.size();
  for (const auto &node : topic_nodes) {
    result_string_length += node.size() + 1; // (+1 for the delimiter character)
  }

  // concatenate the string with a '/' as delimiter
  std::string result;
  result.reserve(result_string_length);
  for (const auto &node : topic_nodes) {
    result.append(node);
    result.push_back('/');
  }
  result.append(topic_name);

  // add the new string to our cache
  {
    std::lock_guard lock(cache_mutex);
    cache[hash] = result;
  }

  return result;
}

} // namespace pc::publisher
