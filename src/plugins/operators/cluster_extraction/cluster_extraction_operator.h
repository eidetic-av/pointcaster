#pragma once

#include "cluster_extraction_config.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <plugins/backend/backend_plugin.h>
#include <plugins/operators/operator_plugin.h>

namespace pc::operators {

class ClusterExtractionOperator final : public OperatorPlugin {
public:
  explicit ClusterExtractionOperator(
      Corrade::PluginManager::AbstractManager &manager,
      Corrade::Containers::StringView plugin)
      : OperatorPlugin(manager, plugin) {}

  ~ClusterExtractionOperator() override {}

  ClusterExtractionOperator(const ClusterExtractionOperator &) = delete;
  ClusterExtractionOperator &
  operator=(const ClusterExtractionOperator &) = delete;
  ClusterExtractionOperator(ClusterExtractionOperator &&) = delete;
  ClusterExtractionOperator &operator=(ClusterExtractionOperator &&) = delete;

  void init(OperatorHost *host,
            Corrade::PluginManager::Manager<backend::BackendPlugin>
                &backend_plugin_manager) override;

  std::shared_ptr<PointCloud> process(const PointCloud &input) override;

  void on_config_field_changed(std::string_view path = "") override;

  const ClusterExtractionConfiguration &config() const {
    return std::get<ClusterExtractionConfiguration>(config_variant());
  }

private:
};

} // namespace pc::operators