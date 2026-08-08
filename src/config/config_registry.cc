#include "config_registry.h"

#include <core/logger/logger.h>

#include <algorithm>

namespace pc {

void ConfigRegistry::register_field(std::string path, Field field) {
  std::unique_lock lock(_mutex);
  _fields.insert_or_assign(std::move(path), std::move(field));
}

void ConfigRegistry::unregister(std::string_view prefix) {
  std::unique_lock lock(_mutex);
  std::erase_if(_fields,
                [&](const auto &kv) { return kv.first.starts_with(prefix); });
}

void ConfigRegistry::clear() {
  std::unique_lock lock(_mutex);
  _fields.clear();
}

bool ConfigRegistry::set(std::string_view path, ConfigValue value) {
  std::function<void(ConfigValue)> setter;
  {
    std::shared_lock lock(_mutex);
    auto it = _fields.find(std::string(path));
    if (it == _fields.end()) {
      return false;
    }
    setter = it->second.set;
  }
  setter(std::move(value));
  notify(path);
  return true;
}

std::optional<ConfigValue> ConfigRegistry::get(std::string_view path) const {
  std::shared_lock lock(_mutex);
  auto it = _fields.find(std::string(path));
  if (it == _fields.end()) return std::nullopt;
  return it->second.get();
}

void ConfigRegistry::on_change(std::string_view prefix, ChangeCallback cb) {
  std::scoped_lock lock(_mutex);
  _subs.emplace_back(std::string(prefix), std::move(cb));
}

void ConfigRegistry::remove_subscriptions(std::string_view prefix) {
  std::scoped_lock lock(_mutex);
  std::erase_if(_subs,
                [&](const auto &sub) { return sub.first.starts_with(prefix); });
}

void ConfigRegistry::notify(std::string_view path) {
  std::vector<ChangeCallback> matching;
  {
    std::shared_lock lock(_mutex);
    for (const auto &[prefix, cb] : _subs) {
      if (path.starts_with(prefix)) matching.push_back(cb);
    }
  }
  for (auto &cb : matching) cb(path);
}

} // namespace pc