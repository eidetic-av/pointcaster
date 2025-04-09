#pragma once

#include <concurrentqueue/concurrentqueue.h>
#include <memory>
#include <mutex>
#include <tweeny/tweeny.h>
#include <unordered_map>
#include <utility>
#include <variant>

namespace pc::tween {

using TweenVariant = std::variant<tweeny::tween<int>, tweeny::tween<float>>;

class TweenManager {
public:
  static std::shared_ptr<TweenManager> &instance() {
    if (!_instance) create();
    return _instance;
  }

  static void create() {
    std::call_once(_instantiated, []() {
      _instance = std::shared_ptr<TweenManager>(new TweenManager());
    });
  }

  TweenManager(const TweenManager &) = delete;
  TweenManager &operator=(const TweenManager &) = delete;

  ~TweenManager();

  ///

  void tick(int delta_ms);

  template <typename T> void add(const std::string &id, T &&tween) {
    _newly_added_tweens.enqueue({id, TweenVariant(std::forward<T>(tween))});
  }

  void remove(const std::string &id) { _tweens_to_remove.enqueue(id); }

  bool has(const std::string &id) const { return _tween_list.count(id) > 0; }
  TweenVariant get(const std::string &id) const { return _tween_list.at(id); }

private:
  static std::shared_ptr<TweenManager> _instance;
  static std::once_flag _instantiated;

  explicit TweenManager();

  ///
  std::unordered_map<std::string, TweenVariant> _tween_list;
  moodycamel::ConcurrentQueue<std::pair<std::string, TweenVariant>> _newly_added_tweens;
  moodycamel::ConcurrentQueue<std::string> _tweens_to_remove;
};

} // namespace pc::tween
