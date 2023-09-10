#include "tween_manager.h"
#include "../logger.h"

namespace pc::tween {

std::shared_ptr<TweenManager> TweenManager::_instance;
std::mutex TweenManager::_mutex;

TweenManager::TweenManager() {}

TweenManager::~TweenManager() {}

struct TweenStepper {
  int delta_ms;

  template <typename T> bool operator()(tweeny::tween<T> &tween) const {
    tween.step(delta_ms);
    return tween.progress() >= 1.0f;
  }
};

void TweenManager::tick(int delta_ms) {
  // pop any requested removals since last tick
  std::string tween_to_remove;
  while (_tweens_to_remove.try_dequeue(tween_to_remove)) {
    _tween_list.erase(tween_to_remove);
  }

  // pop tweens added since last tick
  std::pair<std::string, TweenVariant> newly_added_tween;
  while (_newly_added_tweens.try_dequeue(newly_added_tween)) {
    const auto &[id, new_tween] = newly_added_tween;
    _tween_list[id] = new_tween;
  }

  // step over tween values to advance their timeline,
  // and delete ones that are finished
  TweenStepper stepper{delta_ms};
  for (auto it = _tween_list.begin(); it != _tween_list.end();) {
    bool finished = std::visit(stepper, it->second);
    if (finished) it = _tween_list.erase(it);
    else it++;
  }
}

} // namespace pc::tween
