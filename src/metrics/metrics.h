#pragma once

#include "prometheus_server.h"

namespace pc::metrics {
template <typename T>
inline void
set_gauge(std::string_view metric_name, T &&value,
          std::initializer_list<PrometheusServer::LabelPairView> labels = {}) {
  PrometheusServer::set_gauge(metric_name, std::forward<T>(value), labels);
}

template <typename T>
inline void set_counter(
    std::string_view metric_name, T &&total,
    std::initializer_list<PrometheusServer::LabelPairView> labels = {}) {
  PrometheusServer::set_counter(metric_name, std::forward<T>(total), labels);
}

template <typename T>
inline void observe_duration(
    std::string_view metric_name, T &&milliseconds,
    std::initializer_list<PrometheusServer::LabelPairView> labels = {}) {
  PrometheusServer::observe_duration(metric_name, std::forward<T>(milliseconds),
                                     labels);
}
} // namespace pc::metrics
