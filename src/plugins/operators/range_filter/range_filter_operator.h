#pragma once

#include "range_filter_config.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <plugins/backend/backend_plugin.h>
#include <plugins/operators/operator_plugin.h>

namespace pc::operators {

class RangeFilterOperator final : public OperatorPlugin {
public:
  explicit RangeFilterOperator(Corrade::PluginManager::AbstractManager &manager,
                               Corrade::Containers::StringView plugin)
      : OperatorPlugin(manager, plugin) {}

  ~RangeFilterOperator() override {}

  RangeFilterOperator(const RangeFilterOperator &) = delete;
  RangeFilterOperator &operator=(const RangeFilterOperator &) = delete;
  RangeFilterOperator(RangeFilterOperator &&) = delete;
  RangeFilterOperator &operator=(RangeFilterOperator &&) = delete;

  void init(OperatorHost *host,
            Corrade::PluginManager::Manager<backend::BackendPlugin>
                &backend_plugin_manager) override;

  PipelineFramePtr process(PipelineFramePtr input) override;

  const RangeFilterConfiguration &config() const {
    return std::get<RangeFilterConfiguration>(config_variant());
  }
};

} // namespace pc::operators
