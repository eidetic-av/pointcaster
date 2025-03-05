#pragma once
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace pc::operators {

using uid = unsigned long int;

inline std::unordered_map<uid, std::string> operator_friendly_names;
inline std::unordered_map<std::string, uid> operator_ids_from_friendly_names;

template <typename T>
inline void set_operator_friendly_name(uid id, T&& name) {
    std::string name_str(std::forward<T>(name));
    operator_friendly_names[id] = name_str;
    operator_ids_from_friendly_names[name_str] = id;
}

inline std::string get_operator_friendly_name(uid id) {
  auto it = operator_friendly_names.find(id);
  return (it != operator_friendly_names.end()) ? it->second : "";
}

inline std::string get_operator_friendly_name(uid id,
                                              std::string_view fallback) {
  auto it = operator_friendly_names.find(id);
  return (it != operator_friendly_names.end()) ? it->second
                                               : std::string(fallback);
}

inline std::string get_operator_friendly_name(std::string_view id_str) {
  try {
    uid id = std::stoul(std::string{id_str});
    auto it = operator_friendly_names.find(id);
    return (it != operator_friendly_names.end()) ? it->second : "";
  } catch (const std::exception &) { return ""; }
}

inline const char* get_operator_friendly_name_cstr(uid id) {
    static thread_local std::string tls;
    tls = get_operator_friendly_name(id);
    return tls.c_str();
}

inline std::optional<uid> get_operator_id(std::string_view name) {
    auto it = operator_ids_from_friendly_names.find(std::string{name});
    if (it != operator_ids_from_friendly_names.end()) {
        return it->second;
    }
    return std::nullopt;
}

inline bool operator_friendly_name_exists(std::string_view name) {
    return operator_ids_from_friendly_names.find(std::string{name}) != operator_ids_from_friendly_names.end();
}

} // namespace pc::operators
