#pragma once

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <k4a/k4a.hpp>
#include <atomic>

namespace pc::devices {

class AbstractAzureKinectBodyTracking : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.k4abt/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    return {Corrade::InPlaceInit, {"plugins/k4abt"}};
  }

  explicit AbstractAzureKinectBodyTracking(
      Corrade::PluginManager::AbstractManager &manager,
      Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {}

  std::atomic_bool _stop_requested = false;

  virtual void init(const k4a::calibration &calibration) = 0;

  virtual void enqueue_capture(const k4a::capture &capture) const = 0;
  virtual void track_bodies() = 0;

};

} // namespace pc::analysis
