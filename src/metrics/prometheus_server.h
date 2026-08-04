#pragma once

#include <concurrentqueue/moodycamel/blockingconcurrentqueue.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#if defined(METRICS_DLL)
#define METRICS_API __declspec(dllexport)
#else
#define METRICS_API __declspec(dllimport)
#endif
#else
#define METRICS_API
#endif

namespace prometheus {
class Registry;
}

namespace pc::metrics {

class METRICS_API PrometheusServer {
public:
  static void initialise();
  static void set_enabled(bool enabled, std::string_view bind_address);
  static bool is_enabled();
  static std::shared_ptr<prometheus::Registry> registry();

  using LabelPairView = std::pair<std::string_view, std::string_view>;

  enum class MetricKind { Gauge, Counter, Histogram };

  // a value that moves in both directions
  template <typename T>
  static void set_gauge(std::string_view metric_name, T &&value,
                        std::initializer_list<LabelPairView> labels = {}) {
    enqueue(MetricKind::Gauge, metric_name, static_cast<double>(value), labels);
  }

  // a monotonic ascending value
  template <typename T>
  static void set_counter(std::string_view metric_name, T &&total,
                          std::initializer_list<LabelPairView> labels = {}) {
    enqueue(MetricKind::Counter, metric_name, static_cast<double>(total),
            labels);
  }

  // a duration in milliseconds
  template <typename T>
  static void
  observe_duration(std::string_view metric_name, T &&milliseconds,
                   std::initializer_list<LabelPairView> labels = {}) {
    enqueue(MetricKind::Histogram, metric_name,
            static_cast<double>(milliseconds), labels);
  }

private:
  struct MetricUpdate {
    MetricKind kind = MetricKind::Gauge;
    std::string metric_name;
    double value = 0.0;
    std::vector<std::pair<std::string, std::string>> labels; // key,value
  };

  static void enqueue(MetricKind kind, std::string_view metric_name,
                      double value,
                      std::initializer_list<LabelPairView> labels) {
    MetricUpdate update{.kind = kind,
                        .metric_name = std::string(metric_name),
                        .value = value,
                        .labels{}};
    for (const auto &[k, v] : labels) {
      update.labels.emplace_back(std::string(k), std::string(v));
    }

    _update_queue.enqueue(std::move(update));
  }

  static moodycamel::BlockingConcurrentQueue<MetricUpdate> _update_queue;
};

} // namespace pc::metrics
