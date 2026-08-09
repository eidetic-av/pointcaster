#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <rfl/internal/is_skip.hpp>
#include <type_traits>

namespace pc {

// for wrapping operator fields to make them writeable from
// inside the operator's own process
template <class T> class Output {
  static_assert(std::is_arithmetic_v<T>, "Output only holds arithmetic types");

public:
  using Type = T;
  using ReflectionType = std::optional<T>;
  static constexpr bool skip_serialization_ = true;
  static constexpr bool skip_deserialization_ = true;

  Output() : _state(std::make_shared<ValueState>(T{})) {}
  Output(T value) : _state(std::make_shared<ValueState>(value)) {}
  Output(const ReflectionType &value)
      : _state(std::make_shared<ValueState>(value ? *value : T{})) {}

  void set(T value) const {
    if (_state->value.exchange(value, std::memory_order_relaxed) == value) {
      return;
    }
    if (auto notify = _state->notify.load(std::memory_order_acquire)) {
      (*notify)();
    }
  }

  T value() const { return _state->value.load(std::memory_order_relaxed); }

  ReflectionType reflection() const { return value(); }

  void bind(std::function<void()> on_change) const {
    _state->notify.store(
        std::make_shared<const std::function<void()>>(std::move(on_change)),
        std::memory_order_release);
  }

  friend bool operator==(const Output &a, const Output &b) {
    return a.value() == b.value();
  }

private:
  struct ValueState {
    std::atomic<T> value;
    std::atomic<std::shared_ptr<const std::function<void()>>> notify;
    explicit ValueState(T initial) : value(initial) {}
  };

  std::shared_ptr<ValueState> _state;
};

template <class T> struct is_output : std::false_type {};
template <class T> struct is_output<Output<T>> : std::true_type {};
template <class T> inline constexpr bool is_output_v = is_output<T>::value;

} // namespace pc

// rfl picks what to leave out of serialization through this trait...
// its like using the rfl::Skip type
namespace rfl::internal {
template <class T> class is_skip<pc::Output<T>> : public std::true_type {};
} // namespace rfl::internal
