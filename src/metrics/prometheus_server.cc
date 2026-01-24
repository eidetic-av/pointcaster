#include "prometheus_server.h"
#include <app_settings/app_settings.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <optional>
#include <print>
#include <qobject.h>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <CivetServer.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/registry.h>

#include <core/logger/logger.h>

// singleton instance that lives in this TU
namespace {

struct PrometheusServerHost {
  std::shared_ptr<prometheus::Registry> registry;
  std::string bind_address;
  std::unique_ptr<prometheus::Exposer> exposer;

  explicit PrometheusServerHost(std::string_view addr)
      : registry(std::make_shared<prometheus::Registry>()), bind_address(addr),
        exposer(std::make_unique<prometheus::Exposer>(bind_address)) {
    exposer->RegisterCollectable(registry);
    pc::logger->info("Serving prometheus metrics on http://{}", bind_address);

    // // to test pulling metrics:
    // std::thread([]() {
    //   static thread_local std::mt19937_64 rng{std::random_device{}()};
    //   static thread_local std::uniform_int_distribution<int> dist(45, 60);
    //   while (true) {
    //     pc::metrics::set_gauge("pointcaster_session_frames_per_second",
    //                            dist(rng), {{"session_name", "session_0"}});
    //     using namespace std::chrono_literals;
    //     std::this_thread::sleep_for(50ms);
    //   }
    // }).detach();
  }
};

std::mutex server_host_access;
std::optional<PrometheusServerHost> server_host;

std::jthread server_control_thread;
std::jthread metrics_update_thread;

using GaugeFamily = prometheus::Family<prometheus::Gauge>;

static std::string make_series_key(
    std::string_view metric_name,
    const std::vector<std::pair<std::string, std::string>> &labels_sorted) {

  std::string key;
  key.reserve(metric_name.size() + 16 + labels_sorted.size() * 24);

  key.append(metric_name);
  key.push_back('{');

  bool first = true;
  for (const auto &[k, v] : labels_sorted) {
    if (!first) key.push_back(',');
    first = false;
    key.append(k);
    key.push_back('=');
    key.append(v);
  }

  key.push_back('}');
  return key;
}

static std::vector<std::pair<std::string, std::string>>
canonicalise_labels(std::vector<std::pair<std::string, std::string>> labels) {
  std::sort(labels.begin(), labels.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  return labels;
}

struct GaugeCache {
  std::weak_ptr<prometheus::Registry> cached_registry;

  // metric name -> family pointer (pointer stable for lifetime of registry)
  std::unordered_map<std::string, GaugeFamily *> family_by_metric_name;

  // metric+labels -> gauge pointer
  std::unordered_map<std::string, prometheus::Gauge *> gauge_by_series_key;

  void reset_for_registry(const std::shared_ptr<prometheus::Registry> &reg) {
    family_by_metric_name.clear();
    gauge_by_series_key.clear();
    cached_registry = reg;
  }
};

struct PendingGaugeUpdate {
  double value = 0.0;
  std::string metric_name; // for convenience
  std::vector<std::pair<std::string, std::string>> labels_sorted;
};

} // namespace

namespace pc::metrics {

moodycamel::BlockingConcurrentQueue<PrometheusServer::GaugeUpdate>
    PrometheusServer::_gauge_update_queue;

void PrometheusServer::initialise() {
  using namespace std::chrono;
  using namespace std::chrono_literals;

  metrics_update_thread = std::jthread([](std::stop_token st) {
    using namespace std::chrono;
    using namespace std::chrono_literals;

    GaugeCache gauge_cache;

    // series_key -> latest update
    std::unordered_map<std::string, PendingGaugeUpdate> latest_by_series_key;
    latest_by_series_key.reserve(256);

    auto next_tick = steady_clock::now() + 1s;

    GaugeUpdate update;

    const auto flush_latest = [&]() {
      if (latest_by_series_key.empty()) return;

      auto registry = PrometheusServer::registry();
      if (!registry) {
        // server disabled... drop pending updates
        latest_by_series_key.clear();
        gauge_cache.family_by_metric_name.clear();
        gauge_cache.gauge_by_series_key.clear();
        gauge_cache.cached_registry.reset();
        return;
      }

      // invalidate cache if registry instance changed (server recreated)
      if (gauge_cache.cached_registry.lock() != registry) {
        gauge_cache.reset_for_registry(registry);
      }

      for (auto &[series_key, pending] : latest_by_series_key) {
        // family per metric
        GaugeFamily *family = nullptr;
        if (auto it =
                gauge_cache.family_by_metric_name.find(pending.metric_name);
            it != gauge_cache.family_by_metric_name.end()) {
          family = it->second;
        } else {
          auto &family_ref = prometheus::BuildGauge()
                                 .Name(pending.metric_name)
                                 .Help("auto-created gauge")
                                 .Register(*registry);
          family = &family_ref;
          gauge_cache.family_by_metric_name.emplace(pending.metric_name,
                                                    family);
        }

        // gauge per (metric+labels)
        prometheus::Gauge *gauge = nullptr;
        if (auto it = gauge_cache.gauge_by_series_key.find(series_key);
            it != gauge_cache.gauge_by_series_key.end()) {
          gauge = it->second;
        } else {
          prometheus::Labels label_map;
          for (const auto &[k, v] : pending.labels_sorted) {
            label_map.emplace(k, v);
          }

          auto &gauge_ref = family->Add(std::move(label_map));
          gauge = &gauge_ref;
          gauge_cache.gauge_by_series_key.emplace(series_key, gauge);
        }

        gauge->Set(pending.value);
      }

      latest_by_series_key.clear();
    };

    while (!st.stop_requested()) {
      const auto now = steady_clock::now();

      if (now >= next_tick) {
        flush_latest();
        next_tick = steady_clock::now() + 1s;
        continue;
      }

      const auto remaining = duration_cast<microseconds>(next_tick - now);

      // wait until either an update arrives or we hit the next 1s tick
      if (PrometheusServer::_gauge_update_queue.wait_dequeue_timed(update,
                                                                   remaining)) {

        {
          auto labels_sorted = canonicalise_labels(std::move(update.labels));
          const std::string series_key =
              make_series_key(update.metric_name, labels_sorted);

          latest_by_series_key[series_key] = PendingGaugeUpdate{
              .value = update.value,
              .metric_name = std::move(update.metric_name),
              .labels_sorted = std::move(labels_sorted),
          };
        }

        while (PrometheusServer::_gauge_update_queue.try_dequeue(update)) {
          auto labels_sorted = canonicalise_labels(std::move(update.labels));
          const std::string series_key =
              make_series_key(update.metric_name, labels_sorted);

          latest_by_series_key[series_key] = PendingGaugeUpdate{
              .value = update.value,
              .metric_name = std::move(update.metric_name),
              .labels_sorted = std::move(labels_sorted),
          };
        }
      }
    }

    // final flush on shutdown
    flush_latest();
  });

  auto settings = AppSettings::instance();

  const auto apply_prometheus_settings = [settings] {
    const bool enabled = settings->enablePrometheusMetrics();
    const std::string address = settings->prometheusAddress().toStdString();

    // if set_enabled causes server construction/destruction it can be
    // expensive, so just throw it onto another thread
    server_control_thread.request_stop();
    server_control_thread = std::jthread([enabled, address] {
      PrometheusServer::set_enabled(enabled, address);
    });
  };

  QObject::connect(settings, &AppSettings::enablePrometheusMetricsChanged,
                   settings, apply_prometheus_settings);
  QObject::connect(settings, &AppSettings::prometheusAddressChanged, settings,
                   apply_prometheus_settings);

  apply_prometheus_settings();
}

void PrometheusServer::set_enabled(bool enabled,
                                   std::string_view bind_address) {
  std::scoped_lock lock(server_host_access);

  if (!enabled) {
    server_host.reset();
    pc::logger->info("Disabled prometheus endpoint");
    return;
  }

  // already running on same address
  if (server_host && server_host->bind_address == bind_address) return;

  try {
    server_host.emplace(bind_address);
  } catch (const CivetException &e) {
    pc::logger->error("Failed to create prometheus server: {}", e.what());
  }
}

bool PrometheusServer::is_enabled() {
  std::scoped_lock lock(server_host_access);
  return server_host.has_value();
}

std::shared_ptr<prometheus::Registry> PrometheusServer::registry() {
  std::scoped_lock lock(server_host_access);
  return server_host ? server_host->registry : nullptr;
}

} // namespace pc::metrics
