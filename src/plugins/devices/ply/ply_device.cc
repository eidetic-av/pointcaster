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

void PlyDevice::init(Workspace &workspace) {
  _workspace = &workspace;

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

  auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.active && !config.file.path.empty()) {
    load(config.file.path);
  }

  _tick_thread = std::jthread([this](std::stop_token stop) {
    using clock = std::chrono::steady_clock;
    auto last = clock::now();
    while (!stop.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      auto now = clock::now();
      float dt = std::chrono::duration<float>(now - last).count();
      last = now;
      tick(dt);
    }
  });
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
    return load_directory(path_str);
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
  // _status = DeviceStatus::Loaded;
  apply_transform();
  return true;
}

bool PlyDevice::load_directory(const std::filesystem::path &dir) {
  auto &config = std::get<PlyDeviceConfiguration>(_config);
  auto &seq = config.sequence;

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
  load(config.file.path);
}

void PlyDevice::tick(float delta_time) {
  std::lock_guard lock(_device_mutex);
  if (!_sequence_loader) return;

  auto config = std::get<PlyDeviceConfiguration>(_config);
  auto &seq = config.sequence;

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

  auto frame = _sequence_loader->get_frame(static_cast<size_t>(_current_frame));
  if (frame) {
    _input_cloud = std::move(frame);
    apply_transform();
  }
}

size_t PlyDevice::frame_count() const {
  return _sequence_loader ? _sequence_loader->frame_count() : 1;
}

void PlyDevice::on_config_field_changed(std::string_view path) {
  std::unique_lock lock(_device_mutex);
  auto &config = std::get<PlyDeviceConfiguration>(_config);

  // file path changed... reload
  if (path.find("file") != std::string_view::npos) {
    if (config.file.path != _loaded_file_path) {
      lock.unlock();
      if (load(config.file.path)) {
        lock.lock();
        _loaded_file_path = config.file.path;
      } else {
        lock.lock();
        config.file.path = _loaded_file_path;
      }
      return;
    }
  }

  // sequence config changes
  if (_sequence_loader && path.find("sequence") != std::string_view::npos) {
    auto &seq = config.sequence;

    if (path.find("buffer_capacity") != std::string_view::npos ||
        path.find("prefetch_ahead") != std::string_view::npos) {
      _sequence_loader->invalidate();
      return;
    }

    // scrub...
    if (path.find("current_frame") != std::string_view::npos) {
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

std::shared_ptr<PointCloud> PlyDevice::point_cloud() {
  return _current_point_cloud.load(std::memory_order_acquire);
}

void PlyDevice::apply_transform() {
  if (!_input_cloud) return;

  const auto config = std::get<PlyDeviceConfiguration>(_config);
  const auto point_count = _input_cloud->size();

  auto output = std::make_shared<PointCloud>();
  output->resize(point_count);

  backend::BackendPlugin *backend = _cpu_backend.get();

  if (backend) {
    backend->transform_point_cloud(*_input_cloud, output, config.transform,
                                   config.color);
  }

  for (const auto &[operator_plugin, operator_config] :
       std::views::zip(operators, config.operators)) {
    operator_plugin->process(*output, *output, operator_config);
  }

  std::shared_ptr<std::vector<std::byte>> render_buffer;
  if (config.render && backend) {
    const auto final_count = output->size();
    render_buffer = std::make_shared<std::vector<std::byte>>(final_count * 16);
    backend->pack_render_buffer(*output, *render_buffer);
  }

  _current_point_cloud.store(std::move(output), std::memory_order_release);

  if (render_buffer) {
    _latest_render_data.store(std::move(render_buffer),
                              std::memory_order_release);
  }
  notify_point_cloud_updated();
}

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(PlyDevice, pc::devices::PlyDevice,
                        "net.pointcaster.DevicePlugin/1.0")