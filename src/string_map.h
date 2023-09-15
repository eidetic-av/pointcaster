#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace pc {

// allow hashing to be equivalent between std::strings and std::string_views
struct sv_hasher {
  using is_transparent = void;
  std::size_t operator()(const std::string &str) const {
    return std::hash<std::string>{}(str);
  }
  std::size_t operator()(std::string_view sv) const {
    return std::hash<std::string_view>{}(sv);
  }
};

// allow equality comparisons between std::strings and std::string_views
struct sv_equality {
  using is_transparent = void;
  bool operator()(const std::string &lhs, const std::string &rhs) const {
    return lhs == rhs;
  }
  bool operator()(const std::string &lhs, std::string_view rhs) const {
    return lhs == rhs;
  }
  bool operator()(std::string_view lhs, const std::string &rhs) const {
    return lhs == rhs;
  }
  bool operator()(std::string_view lhs, std::string_view rhs) const {
    return lhs == rhs;
  }
};

// just a simple wrapper to a std::unordered_map that allows lookup via both
// std::string and std::string_view
template <typename ValueType> class string_map {
private:
  using MapType =
      std::unordered_map<std::string, ValueType, sv_hasher, sv_equality>;
  MapType _inner_map;

public:
  ValueType &at(const std::string &key) { return _inner_map.at(key); }

  ValueType &at(std::string_view key) {
    auto it = _inner_map.find(key);
    if (it == _inner_map.end())
      throw std::out_of_range("Key not found");
    return it->second;
  }

  bool contains(const std::string &key) const {
    return _inner_map.contains(key);
  }

  bool contains(std::string_view key) { return _inner_map.contains(key); }

  void erase(const std::string &key) { _inner_map.erase(key); }

  void erase(std::string_view key) { _inner_map.erase(std::string(key)); }

  std::pair<typename MapType::iterator, bool> emplace(const std::string &key,
                                                      const ValueType &value) {
    return _inner_map.emplace(key, value);
  }

  std::pair<typename MapType::iterator, bool> emplace(const std::string &key,
                                                      ValueType &&value) {
    return _inner_map.emplace(key, std::move(value));
  }

  std::pair<typename MapType::iterator, bool> emplace(std::string_view key,
                                                      const ValueType &value) {
    return _inner_map.emplace(std::string(key), value);
  }

  std::pair<typename MapType::iterator, bool> emplace(std::string_view key,
                                                      ValueType &&value) {
    return _inner_map.emplace(std::string(key), std::move(value));
  }

  template <typename... Args>
  std::pair<typename MapType::iterator, bool>
  emplace(std::pair<std::string, ValueType> &&pair) {
    return _inner_map.emplace(std::move(pair));
  }

  template <typename... Args>
  std::pair<typename MapType::iterator, bool>
  emplace(std::pair<std::string_view, ValueType> &&pair) {
    return _inner_map.emplace(
        std::make_pair(std::string(pair.first), std::move(pair.second)));
  }

  template <typename... Args>
  std::pair<typename MapType::iterator, bool>
  try_emplace(const std::string &key, Args &&...args) {
    return _inner_map.try_emplace(key, std::forward<Args>(args)...);
  }

  template <typename... Args>
  std::pair<typename MapType::iterator, bool> try_emplace(std::string_view key,
                                                          Args &&...args) {
    return _inner_map.try_emplace(std::string(key),
                                  std::forward<Args>(args)...);
  }

  ValueType &operator[](const std::string &key) { return _inner_map[key]; }

  ValueType &operator[](std::string_view key) {
    auto it = _inner_map.find(key);
    if (it != _inner_map.end()) {
      return it->second;
    } else {
      // if the key is not found, we allocate and insert
      return _inner_map[std::string(key)];
    }
  }
};

} // namespace pc
