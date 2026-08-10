#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <pointcaster/point_cloud.h>
#include <rfl/internal/is_skip.hpp>
#include <type_traits>
#include <variant>

namespace pc {

template <class T>
concept OutputScalar = std::is_arithmetic_v<T>;

template <class T>
concept OutputStream = is_cloud_stream_v<T>;

template <class T>
concept OutputValue = OutputScalar<T> || OutputStream<T>;

// for wrapping operator fields to make them writeable from
// inside the operator's own process
template <class T> class Output {
  static_assert(OutputValue<T>,
                "Output only holds arithmetic types or cloud streams");

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
    if constexpr (OutputScalar<T>) {
      if (value) set(*value);
    }
  }

  void set(T value) const {
    const auto previous_value =
        _state->value.exchange(value, std::memory_order_relaxed);
    if (previous_value == value) return;
    if (auto notify = _state->notify.load(std::memory_order_acquire)) {
      (*notify)();
    }
  }

  T value() const { return _state->value.load(std::memory_order_relaxed); }

  // this exists to satisfy rfl as a skipped wrapper
  ReflectionType reflection() const {
    if constexpr (OutputScalar<T>) {
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
  struct ValueState {
    // atomic<shared_ptr> for streams, a plain atomic for scalars
    std::atomic<T> value;
    std::atomic<std::shared_ptr<const std::function<void()>>> notify;
    explicit ValueState(T initial) : value(std::move(initial)) {}
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
