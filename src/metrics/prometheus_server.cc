#include "prometheus_server.h"
#include <app_settings/app_settings.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <optional>
#include <print>
#include <qobject.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <CivetServer.h>
#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
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
    pc::logger()->info("Serving prometheus metrics on http://{}", bind_address);
  }
};

std::mutex server_host_access;
std::optional<PrometheusServerHost> server_host;

std::jthread server_control_thread;
std::jthread metrics_update_thread;

using GaugeFamily = prometheus::Family<prometheus::Gauge>;
using CounterFamily = prometheus::Family<prometheus::Counter>;
using HistogramFamily = prometheus::Family<prometheus::Histogram>;

using MetricKind = pc::metrics::PrometheusServer::MetricKind;

const prometheus::Histogram::BucketBoundaries duration_buckets_ms{
    0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000};

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

struct MetricCache {
  std::weak_ptr<prometheus::Registry> cached_registry;

  // metric name -> family pointer
  std::unordered_map<std::string, GaugeFamily *> gauge_families;
  std::unordered_map<std::string, CounterFamily *> counter_families;
  std::unordered_map<std::string, HistogramFamily *> histogram_families;

  // metric+labels -> instance pointer
  std::unordered_map<std::string, prometheus::Gauge *> gauges;
  std::unordered_map<std::string, prometheus::Counter *> counters;
  std::unordered_map<std::string, prometheus::Histogram *> histograms;

  // the last total published per counter series
  std::unordered_map<std::string, double> counter_totals;

  void reset_for_registry(const std::shared_ptr<prometheus::Registry> &reg) {
    gauge_families.clear();
    counter_families.clear();
    histogram_families.clear();
    gauges.clear();
    counters.clear();
    histograms.clear();
    counter_totals.clear();
    cached_registry = reg;
  }
};

struct PendingUpdate {
  MetricKind kind = MetricKind::Gauge;
  double value = 0.0;
  std::vector<double> observations;
  std::string metric_name;
  std::vector<std::pair<std::string, std::string>> labels_sorted;
};

} // namespace

namespace pc::metrics {

moodycamel::BlockingConcurrentQueue<PrometheusServer::MetricUpdate>
    PrometheusServer::_update_queue;

void PrometheusServer::initialise() {
  using namespace std::chrono;
  using namespace std::chrono_literals;

  metrics_update_thread = std::jthread([](std::stop_token st) {
    using namespace std::chrono;
    using namespace std::chrono_literals;

    MetricCache cache;

    // series_key -> pending update
    std::unordered_map<std::string, PendingUpdate> pending_by_series_key;
    pending_by_series_key.reserve(256);

    auto next_tick = steady_clock::now() + 1s;

    MetricUpdate update;

    const auto label_map = [](const PendingUpdate &pending) {
      prometheus::Labels labels;
      for (const auto &[k, v] : pending.labels_sorted) {
        labels.emplace(k, v);
      }
      return labels;
    };

    const auto flush_latest = [&]() {
      if (pending_by_series_key.empty()) return;

      auto registry = PrometheusServer::registry();
      if (!registry) {
        // server disabled... drop pending updates
        pending_by_series_key.clear();
        cache.reset_for_registry(nullptr);
        return;
      }

      // invalidate cache if registry instance changed (server recreated)
      if (cache.cached_registry.lock() != registry) {
        cache.reset_for_registry(registry);
      }

      for (auto &[series_key, pending] : pending_by_series_key) {
        if (pending.kind == MetricKind::Gauge) {
          auto family = cache.gauge_families.find(pending.metric_name);
          if (family == cache.gauge_families.end()) {
            family = cache.gauge_families
                         .emplace(pending.metric_name,
                                  &prometheus::BuildGauge()
                                       .Name(pending.metric_name)
                                       .Help("auto-created gauge")
                                       .Register(*registry))
                         .first;
          }

          auto gauge = cache.gauges.find(series_key);
          if (gauge == cache.gauges.end()) {
            gauge = cache.gauges
                        .emplace(series_key,
                                 &family->second->Add(label_map(pending)))
                        .first;
          }

          gauge->second->Set(pending.value);

        } else if (pending.kind == MetricKind::Counter) {
          auto family = cache.counter_families.find(pending.metric_name);
          if (family == cache.counter_families.end()) {
            family = cache.counter_families
                         .emplace(pending.metric_name,
                                  &prometheus::BuildCounter()
                                       .Name(pending.metric_name)
                                       .Help("auto-created counter")
                                       .Register(*registry))
                         .first;
          }

          auto counter = cache.counters.find(series_key);
          if (counter == cache.counters.end()) {
            counter = cache.counters
                          .emplace(series_key,
                                   &family->second->Add(label_map(pending)))
                          .first;
          }

          // a total that went backwards means the caller restarted its own
          // count, so the whole new total is the increment
          auto &last_total = cache.counter_totals[series_key];
          const auto increment = pending.value < last_total
                                     ? pending.value
                                     : pending.value - last_total;
          last_total = pending.value;
          if (increment > 0.0) counter->second->Increment(increment);

        } else {
          auto family = cache.histogram_families.find(pending.metric_name);
          if (family == cache.histogram_families.end()) {
            family = cache.histogram_families
                         .emplace(pending.metric_name,
                                  &prometheus::BuildHistogram()
                                       .Name(pending.metric_name)
                                       .Help("auto-created histogram")
                                       .Register(*registry))
                         .first;
          }

          auto histogram = cache.histograms.find(series_key);
          if (histogram == cache.histograms.end()) {
            histogram = cache.histograms
                            .emplace(series_key,
                                     &family->second->Add(label_map(pending),
                                                          duration_buckets_ms))
                            .first;
          }

          for (const auto observation : pending.observations) {
            histogram->second->Observe(observation);
          }
        }
      }

      pending_by_series_key.clear();
    };

    // gauges and counters keep only the newest value per series, histograms
    // accumulate every observation until the next flush
    const auto record = [&](MetricUpdate &&incoming) {
      auto labels_sorted = canonicalise_labels(std::move(incoming.labels));
      const std::string series_key =
          make_series_key(incoming.metric_name, labels_sorted);

      auto &pending = pending_by_series_key[series_key];
      pending.kind = incoming.kind;
      pending.metric_name = std::move(incoming.metric_name);
      pending.labels_sorted = std::move(labels_sorted);

      if (incoming.kind == MetricKind::Histogram) {
        pending.observations.push_back(incoming.value);
      } else {
        pending.value = incoming.value;
      }
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
      if (PrometheusServer::_update_queue.wait_dequeue_timed(update,
                                                             remaining)) {
        record(std::move(update));

        while (PrometheusServer::_update_queue.try_dequeue(update)) {
          record(std::move(update));
        }
      }
    }

    flush_latest();
  });

  auto settings = AppSettings::instance();

  const auto apply_prometheus_settings = [settings] {
    const bool enabled = settings->enablePrometheusMetrics();
    const std::string address = settings->prometheusAddress().toStdString();

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
    if (server_host.has_value()) {
      server_host.reset();
      pc::logger()->info("Stopped prometheus endpoint");
    }
    return;
  }

  if (server_host && server_host->bind_address == bind_address) {
    return;
  }

  try {
    server_host.emplace(bind_address);
  } catch (const CivetException &e) {
    pc::logger()->error("Failed to create prometheus server: {}", e.what());
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
