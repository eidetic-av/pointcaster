#include "ply_device.h"
#include "plugins/devices/device_variants.h"

#include <core/logger/logger.h>
// #include <core/profiling/profiling_zone.h>
#include <cstring>
#include <filesystem>
#include <happly.h>
#include <oneapi/tbb/parallel_for.h>
#include <plugins/backend/cpu/cpu_backend.h>
#include <pointcaster/point_cloud.h>
#include <ranges>
#include <workspace/workspace.h>

namespace pc::devices {

// using pc::profiling::ProfilingZone;

void PlyDevice::init() {
  auto &backend_manager = _workspace->backend_plugin_manager;
  _cpu_backend = backend_manager->instantiate("CpuBackend");

  pc::logger()->trace("PlyDevice created CPU backend");

  using Corrade::PluginManager::LoadState;

  for (const auto &plugin : backend_manager->pluginList()) {
    if (backend_manager->loadState(plugin) & LoadState::NotLoaded) continue;
    if (plugin == "CudaBackend") {
      _cuda_backend = backend_manager->instantiate(plugin);
      pc::logger()->trace("PlyDevice created CUDA backend");
    }
  }
}

PlyDevice::~PlyDevice() {
  _tick_thread.request_stop();
  if (_tick_thread.joinable()) _tick_thread.join();
}

bool PlyDevice::load(std::string_view url) {
  std::lock_guard lock(_device_mutex);
  pc::logger()->trace("Loading ply(s) from: {}", url);

#ifdef _WIN32
  static constexpr std::string file_prefix = "file:///";
#else
  static constexpr std::string file_prefix = "file://";
#endif

  auto path_str = std::string(
      url.starts_with(file_prefix) ? url.substr(file_prefix.size()) : url);

  if (std::filesystem::is_directory(path_str)) {
    if (!load_directory(path_str)) return false;
    _loaded_file_path = std::string(url);
    return true;
  }

  // ── single file mode ──
  _sequence_loader.reset();

  pc::logger()->trace("Loading ply file: {}", path_str);

  std::optional<happly::PLYData> ply_in;

  try {
    // ProfilingZone load_file_zone("PlyDevice::load_file");
    // load_file_zone.text(path_str);
    ply_in.emplace(path_str);
  } catch (const std::runtime_error &e) {
    pc::logger()->error("Exception loading file: {}", e.what());
    // _status = DeviceStatus::Error;
    return false;
  }

  constexpr auto vertex = "vertex";

  const auto x_values = ply_in->getElement(vertex).getProperty<short>("x");
  const auto y_values = ply_in->getElement(vertex).getProperty<short>("y");
  const auto z_values = ply_in->getElement(vertex).getProperty<short>("z");
  const auto r_values =
      ply_in->getElement(vertex).getProperty<unsigned char>("red");
  const auto g_values =
      ply_in->getElement(vertex).getProperty<unsigned char>("green");
  const auto b_values =
      ply_in->getElement(vertex).getProperty<unsigned char>("blue");

  const size_t point_count = x_values.size();

  auto input_cloud = std::make_shared<PointCloud>();
  input_cloud->resize(point_count);

  {
    // ProfilingZone convert_zone("Convert and pack points");
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, point_count),
        [&](const tbb::blocked_range<size_t> &range) {
          for (size_t i = range.begin(), e = range.end(); i < e; ++i) {
            input_cloud->positions[i] = {x_values[i], y_values[i], z_values[i]};
            input_cloud->colors[i] = {r_values[i], g_values[i], b_values[i]};
          }
        });
  }

  _input_cloud = std::move(input_cloud);
  _loaded_file_path = std::string(url);
  // _status = DeviceStatus::Loaded;
  apply_transform();
  return true;
}

bool PlyDevice::load_directory(const std::filesystem::path &dir) {
  auto &config = std::get<PlyDeviceConfiguration>(_config);
  auto &seq = config.sequence.value();

  _input_cloud.reset();
  _sequence_loader.emplace();

  ply::PlySequenceLoader::Config loader_config{
      .buffer_capacity =
          static_cast<size_t>(std::max(8, seq.buffer_capacity.value())),
      .prefetch_ahead =
          static_cast<size_t>(std::max(1, seq.prefetch_ahead.value())),
  };

  if (!_sequence_loader->open(dir, loader_config)) {
    _sequence_loader.reset();
    // _status = DeviceStatus::Error;
    return false;
  }

  _current_frame = 0;
  _frame_accumulator = 0.f;
  // _status = DeviceStatus::Loaded;

  _input_cloud = _sequence_loader->get_frame(0);
  apply_transform();
  return true;
}

void PlyDevice::reload() {
  const auto &config = std::get<PlyDeviceConfiguration>(_config);
  load(config.file.value().path);
}

void PlyDevice::tick(float delta_time) {
  std::lock_guard lock(_device_mutex);
  if (!_sequence_loader) return;

  auto &config = std::get<PlyDeviceConfiguration>(_config);
  auto &seq = config.sequence.value();

  if (!seq.playing.value()) return;

  _frame_accumulator += delta_time * static_cast<float>(seq.frame_rate.value());
  const auto advance = static_cast<int>(_frame_accumulator);
  if (advance == 0) return;
  _frame_accumulator -= static_cast<float>(advance);

  const auto total = static_cast<int>(_sequence_loader->frame_count());
  const auto start = std::clamp(seq.start_frame.value(), 0, total - 1);
  const auto end = (seq.end_frame.value() < 0)
                       ? total - 1
                       : std::clamp(seq.end_frame.value(), start, total - 1);
  const auto range = end - start + 1;

  if (_current_frame < start || _current_frame > end) _current_frame = start;

  auto next = _current_frame + advance;

  if (next > end) {
    if (seq.looping.value()) {
      next = start + (next - start) % range;
    } else {
      next = end;
      seq.playing.set(false);
    }
  }

  if (next == _current_frame) return;
  _current_frame = next;
  seq.current_frame.set(_current_frame);

  _sequence_loader->set_loop(static_cast<size_t>(start),
                             static_cast<size_t>(end));

  if (auto frame =
          _sequence_loader->get_frame(static_cast<size_t>(_current_frame))) {
    _input_cloud = std::move(frame);
    apply_transform();
  }
}

size_t PlyDevice::frame_count() const {
  return _sequence_loader ? _sequence_loader->frame_count() : 1;
}

void PlyDevice::on_config_field_changed(std::string_view path) {
  DevicePlugin::on_config_field_changed(path);

  std::unique_lock lock(_device_mutex);
  auto &config = std::get<PlyDeviceConfiguration>(_config);

  // file path changed... reload
  if (path.find("file") != std::string_view::npos) {
    auto file_config = config.file.value();
    if (file_config.path != _loaded_file_path) {
      lock.unlock();
      if (!load(file_config.path)) {
        lock.lock();
        // rollback
        file_config.path = _loaded_file_path;
        config.file.set(file_config);
      }
      return;
    }
  }

  // sequence config changes
  if (_sequence_loader && path.find("sequence") != std::string_view::npos) {
    if (path.find("buffer_capacity") != std::string_view::npos ||
        path.find("prefetch_ahead") != std::string_view::npos) {
      _sequence_loader->invalidate();
      return;
    }

    // scrub...
    if (path.find("current_frame") != std::string_view::npos) {
      _current_frame = config.sequence.value().current_frame.value();
      const auto &scrub_seq = config.sequence.value();
      const auto total = static_cast<int>(_sequence_loader->frame_count());
      const auto start =
          std::clamp(scrub_seq.start_frame.value(), 0, total - 1);
      const auto end =
          (scrub_seq.end_frame.value() < 0)
              ? total - 1
              : std::clamp(scrub_seq.end_frame.value(), start, total - 1);
      _sequence_loader->set_loop(static_cast<size_t>(start),
                                 static_cast<size_t>(end));
      auto frame =
          _sequence_loader->get_frame(static_cast<size_t>(_current_frame));
      if (frame) {
        _input_cloud = std::move(frame);
        apply_transform();
      }
      return;
    }
  }

  // transform / color / operator changes... trigger re-transform current frame
  if (path.empty() || path.find("transform") != std::string_view::npos ||
      path.find("color") != std::string_view::npos ||
      path.find("operator") != std::string_view::npos ||
      path.find("render") != std::string_view::npos) {
    apply_transform();
  }
}

void PlyDevice::update_config(
    const devices::DeviceConfigurationVariant &config) {
  std::string path_to_load;
  bool need_tick_thread = false;
  {
    std::lock_guard lock(_device_mutex);
    DevicePlugin::update_config(config);
    if (!std::holds_alternative<PlyDeviceConfiguration>(_config)) return;
    const auto &config = std::get<PlyDeviceConfiguration>(_config);
    const auto &file_config = config.file.value();
    if (active() && !file_config.path.empty() &&
        file_config.path != _loaded_file_path) {
      path_to_load = file_config.path;
    }
    need_tick_thread = !_tick_thread.joinable();
  }

  if (!path_to_load.empty()) {
    load(path_to_load);
  }

  if (need_tick_thread) {
    // TODO replace the tick thread with a session or workspace-wide one that we
    // can register with
    _tick_thread = std::jthread([this](std::stop_token stop) {
      using namespace std::chrono;
      auto last = steady_clock::now();
      while (!stop.stop_requested()) {
        std::this_thread::sleep_for(milliseconds(10));
        auto now = steady_clock::now();
        float dt = duration<float>(now - last).count();
        last = now;
        tick(dt);
      }
    });
  }
}

std::shared_ptr<PointCloud> PlyDevice::point_cloud() {
  return _current_point_cloud.load(std::memory_order_acquire);
}

void PlyDevice::apply_transform() {
  if (!_input_cloud) return;

  const auto config = std::get<PlyDeviceConfiguration>(_config);
  const auto point_count = _input_cloud->size();

  // TODO get backend from config
  auto *backend = _cpu_backend.get();
  if (!backend) {
    pc::logger()->error("uninitalised PlyDevice backend");
    return;
  }

  auto transformed_cloud = std::make_shared<PointCloud>();
  transformed_cloud->resize(point_count);

  backend->transform_point_cloud(*_input_cloud, transformed_cloud,
                                 config.transform.value(),
                                 config.color.value());

  feed_operator_pipeline(transformed_cloud);
}

void PlyDevice::on_pipeline_output(std::shared_ptr<PointCloud> processed) {
  const auto config = std::get<PlyDeviceConfiguration>(_config);
  if (rendering() && _cpu_backend) {
    auto buf = std::make_shared<std::vector<std::byte>>(processed->size() * 16);
    _cpu_backend->pack_render_buffer(*processed, *buf);
    _latest_render_data.store(std::move(buf), std::memory_order_release);
  }
  _current_point_cloud.store(std::move(processed), std::memory_order_release);
  notify_point_cloud_updated();
}

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(PlyDevice, pc::devices::PlyDevice,
                        "net.pointcaster.DevicePlugin/1.0")