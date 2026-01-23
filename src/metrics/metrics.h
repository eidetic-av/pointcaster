#pragma once

#include "prometheus_server.h"

namespace pc::metrics {
template <typename T>
inline void
set_gauge(std::string_view metric_name, T &&value,
          std::initializer_list<PrometheusServer::LabelPairView> labels = {}) {
  PrometheusServer::set_gauge(metric_name, std::forward<T>(value), labels);
}
} // namespace pc::metrics