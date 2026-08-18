#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <rfl/internal/is_skip.hpp>
#include <type_traits>
#include <variant>

namespace pc {

template <class T>
concept OutputScalar = std::is_arithmetic_v<T>;

// a box an operator measures, handed out whole rather than as its corners
template <class T>
concept OutputBounds = std::is_same_v<T, position_bounds>;

template <class T>
concept OutputStream = is_cloud_stream_v<T>;

template <class T>
concept OutputValue = OutputScalar<T> || OutputBounds<T> || OutputStream<T>;

// for wrapping operator fields to make them writeable from
// inside the operator's own process
template <class T> class Output {
  static_assert(OutputValue<T>,
                "Output only holds arithmetic types, bounds or cloud streams");

public:
  using Type = T;

  // ReflectionType is the type used by rfl to treat this field as not required
  // for serialization
  using ReflectionType =
      std::optional<std::conditional_t<OutputStream<T>, std::monostate, T>>;
  static constexpr bool skip_serialization_ = true;
  static constexpr bool skip_deserialization_ = true;

  Output() : _state(std::make_shared<ValueState>(T{})) {}
  Output(T value) : _state(std::make_shared<ValueState>(std::move(value))) {}
  Output(const ReflectionType &value) : Output() {
    if constexpr (!OutputStream<T>) {
      if (value) set(*value);
    }
  }

  void set(T value) const {
    const auto previous_value =
        _state->value.exchange(hold(value), std::memory_order_relaxed);
    if (unwrap(previous_value) == value) return;
    if (auto notify = _state->notify.load(std::memory_order_acquire)) {
      (*notify)();
    }
  }

  T value() const {
    return unwrap(_state->value.load(std::memory_order_relaxed));
  }

  // this exists to satisfy rfl as a skipped wrapper
  ReflectionType reflection() const {
    if constexpr (!OutputStream<T>) {
      return value();
    } else {
      return value() ? std::optional(std::monostate{}) : std::nullopt;
    }
  }

  void bind(std::function<void()> on_change) const {
    _state->notify.store(
        std::make_shared<const std::function<void()>>(std::move(on_change)),
        std::memory_order_release);
  }

  friend bool operator==(const Output &a, const Output &b) {
    return a.value() == b.value();
  }

private:
  // std::atomic only lowers to a cpu instruction for types up to a machine
  // word. a position_bounds is sixteen bytes, and gcc turns every read and
  // write of one into an out-of-line libatomic call rather than inlining it,
  // so wide values get boxed and we swap the pointer instead - the same thing
  // ConcurrentContainer does. streams are already an atomic<shared_ptr> and
  // must not be boxed a second time
  static constexpr bool box_value =
      !OutputStream<T> && !std::atomic<T>::is_always_lock_free;

  using Held = std::conditional_t<box_value, std::shared_ptr<const T>, T>;

  static Held hold(T value) {
    if constexpr (box_value)
      return std::make_shared<const T>(std::move(value));
    else
      return value;
  }

  static T unwrap(const Held &held) {
    if constexpr (box_value)
      return held ? *held : T{};
    else
      return held;
  }

  struct ValueState {
    std::atomic<Held> value;
    std::atomic<std::shared_ptr<const std::function<void()>>> notify;
    explicit ValueState(T initial) : value(hold(std::move(initial))) {}
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
