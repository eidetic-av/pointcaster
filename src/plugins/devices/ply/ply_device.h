#pragma once

#include "../device_plugin.h"

#include "plugins/devices/device_status.h"
#include "plugins/devices/ply/ply_device_config.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <string_view>
#include <vector>

// #include <llfio.hpp>
// #include <llfio/v2.0/config.hpp>
// #include <llfio/v2.0/directory_handle.hpp>

namespace pc::devices {

class PlyDevice : public DevicePlugin {
public:
  explicit PlyDevice(Corrade::PluginManager::AbstractManager &manager,
                     Corrade::Containers::StringView plugin)
      : DevicePlugin(manager, plugin) {}

  ~PlyDevice() override {}

  PlyDevice(const PlyDevice &) = delete;
  PlyDevice &operator=(const PlyDevice &) = delete;
  PlyDevice(PlyDevice &&) = delete;
  PlyDevice &operator=(PlyDevice &&) = delete;

  void init(Workspace &workspace) override;

  DeviceStatus status() const override { return _status; };

  std::shared_ptr<PointCloud> point_cloud() override;

  void start() override {};
  void stop() override {};
  void restart() override {};

  void on_config_field_changed(std::string_view path = "") override;

  bool load_file(std::string_view url);

private:
  std::string _loaded_file_path{};
  DeviceStatus _status = DeviceStatus::Unloaded;

  PointCloud _current_point_cloud{{}, {}};

  // // using namespace llfio = LLFIO_V2_NAMESPACE;

  // // _sequence_file_entries is the cache of path metadata for all the files
  // // contained in the sequence
  // std::vector<LLFIO_V2_NAMESPACE::directory_entry> _sequence_file_entries;

  // // _sequence_file_handles is the cache of open file handles for a
  // // wanted subset of the files contained in the sequence. the objects in
  // this
  // // collection are handles to memory mapped files, so they may be loaded in
  // RAM
  // // upon access or they may need to read from disk first depending on
  // current
  // // pressure as determined by the kernel
  // std::vector<LLFIO_V2_NAMESPACE::file_handle> _sequence_file_handles;

  // and _sequence_cloud_buffer is the cache of actual PointCloud frames that we
  // are ensuring will always be available in RAM. _sequence_file_handles above
  // might contain ptrs to the data in RAM, but that access is not garunteed and
  // may have to go to disk. this inner cache is adjacent to the source memory
  // mapped file and will never go to disk when accessed
  std::vector<PointCloud> _sequence_cloud_buffer{128};

  std::atomic_int _current_sequence_frame_index = 0;
};

} // namespace pc::devices