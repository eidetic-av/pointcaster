#pragma once

#include <concurrentqueue/blockingconcurrentqueue.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace prometheus {
class Registry;
}

namespace pc::metrics {

class PrometheusServer {
public:
  static void initialise();
  static void set_enabled(bool enabled, std::string_view bind_address);
  static bool is_enabled();
  static std::shared_ptr<prometheus::Registry> registry();

  using LabelPairView = std::pair<std::string_view, std::string_view>;
  template <typename T>
  static void set_gauge(std::string_view metric_name, T &&value,
                        std::initializer_list<LabelPairView> labels = {}) {
    GaugeUpdate update{.metric_name = std::string(metric_name),
                       .value = static_cast<double>(value),
                       .labels{}};
    for (const auto &[k, v] : labels) {
      update.labels.emplace_back(std::string(k), std::string(v));
    }

    _gauge_update_queue.enqueue(std::move(update));
  }

private:
  struct GaugeUpdate {
    std::string metric_name;
    double value = 0.0;
    std::vector<std::pair<std::string, std::string>> labels; // key,value
  };

  static moodycamel::BlockingConcurrentQueue<GaugeUpdate> _gauge_update_queue;
};

} // namespace pc::metrics
