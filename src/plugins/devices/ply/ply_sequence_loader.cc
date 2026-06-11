#include "ply_sequence_loader.h"

#include <algorithm>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <charconv>
#include <core/logger/logger.h>
// #include <core/profiling/profiling_zone.h>
#include <cstring>
#include <oneapi/tbb/parallel_for.h>
#include <plugins/backend/cpu/cpu_backend.h>

namespace pc::devices::ply {

// TODO replace this with something like happly???
std::optional<HeaderInfo> scan_header(const char *data, size_t length) {
  constexpr std::string_view sentinel = "end_header\n";
  const auto search_len = std::min(length, size_t{4096});
  const auto *search_end = data + search_len;
  const auto *end =
      std::search(data, search_end, sentinel.begin(), sentinel.end());
  if (end == search_end) return std::nullopt;

  HeaderInfo info{};
  info.data_offset = static_cast<size_t>(end - data) + sentinel.size();

  constexpr std::string_view prefix = "element vertex ";
  std::string_view header(data, static_cast<size_t>(end - data));
  auto pos = header.find(prefix);
  if (pos == std::string_view::npos) return std::nullopt;

  auto num_start = pos + prefix.size();
  auto line_end = header.find('\n', num_start);
  auto num_sv = header.substr(num_start, line_end - num_start);
  if (!num_sv.empty() && num_sv.back() == '\r') num_sv.remove_suffix(1);

  auto [ptr, ec] = std::from_chars(num_sv.data(), num_sv.data() + num_sv.size(),
                                   info.vertex_count);
  if (ec != std::errc{}) return std::nullopt;

  return info;
}

std::shared_ptr<PointCloud> load_frame(const std::string &path) {
  //   pc::profiling::ProfilingZone zone("ply::load_frame");

  using namespace boost::interprocess;

  file_mapping mapping;
  try {
    mapping = file_mapping(path.c_str(), boost::interprocess::read_only);
  } catch (const interprocess_exception &e) {
    pc::logger()->error("mmap failed: {} — {}", path, e.what());
    return nullptr;
  }

  mapped_region region(mapping, boost::interprocess::read_only);
  const auto *base = static_cast<const char *>(region.get_address());
  const auto file_size = region.get_size();

  auto header_info = scan_header(base, file_size);
  if (!header_info) {
    pc::logger()->error("bad PLY header: {}", path);
    return nullptr;
  }

  auto [data_offset, vertex_count] = header_info.value();

  if (data_offset + vertex_count * vertex_stride > file_size) {
    pc::logger()->error("truncated PLY: {}", path);
    return nullptr;
  }

  const auto *vertex_data = base + data_offset;

  auto cloud = std::make_shared<PointCloud>();
  cloud->resize(vertex_count);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, vertex_count),
                    [&](const tbb::blocked_range<size_t> &range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        const auto *v = vertex_data + i * vertex_stride;
                        // TODO warning about copying non-full types here...
                        // like we only copy in RGB not A and POS not padding
                        std::memcpy(&cloud->positions[i], v + pos_offset, 6);
                        std::memcpy(&cloud->colors[i], v + color_offset, 3);
                      }
                    });

  return cloud;
}

bool PlySequenceLoader::open(const std::filesystem::path &directory,
                             const Config &config) {
  _config = config;
  _file_paths.clear();

  if (!std::filesystem::is_directory(directory)) {
    pc::logger()->error("not a directory: {}", directory.string());
    return false;
  }

  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() == ".ply")
      _file_paths.emplace_back(entry.path().string());
  }
  std::sort(_file_paths.begin(), _file_paths.end());

  if (_file_paths.empty()) {
    pc::logger()->warn("no .ply files in {}", directory.string());
    return false;
  }

  pc::logger()->info("sequence: {} frames in {}", _file_paths.size(),
                     directory.string());

  _ring.assign(_config.buffer_capacity, nullptr);
  _ring_index.assign(_config.buffer_capacity, npos);

  return true;
}

std::shared_ptr<PointCloud> PlySequenceLoader::get_frame(size_t frame) {
  if (frame >= _file_paths.size()) return nullptr;

  prefetch_from(frame);

  const auto slot = frame % _config.buffer_capacity;

  {
    std::shared_lock lock(_mutex);
    if (_ring_index[slot] == frame && _ring[slot]) return _ring[slot];
  }

  return load_into_slot(frame, slot);
}

void PlySequenceLoader::invalidate() {
  std::unique_lock lock(_mutex);
  std::ranges::fill(_ring_index, npos);
  std::ranges::fill(_ring, nullptr);
  _generation.fetch_add(1, std::memory_order_relaxed);
}

std::shared_ptr<PointCloud> PlySequenceLoader::load_into_slot(size_t frame,
                                                              size_t slot) {
  auto cloud = load_frame(_file_paths[frame]);
  if (!cloud) return nullptr;
  std::unique_lock lock(_mutex);
  _ring[slot] = cloud;
  _ring_index[slot] = frame;
  return cloud;
}

void PlySequenceLoader::set_loop(size_t start, size_t end) {
  _loop_start.store(start, std::memory_order_relaxed);
  _loop_end.store(end, std::memory_order_relaxed);
}

void PlySequenceLoader::prefetch_from(size_t current) {
  auto &pool = backend::CpuBackend::thread_pool;
  const auto gen = _generation.load(std::memory_order_relaxed);

  const auto loop_start = _loop_start.load(std::memory_order_relaxed);
  const auto loop_end = std::min(_loop_end.load(std::memory_order_relaxed),
                                 _file_paths.size() - 1);
  const auto loop_range = loop_end - loop_start + 1;

  for (size_t off = 1; off <= _config.prefetch_ahead; ++off) {
    const auto frame_index = loop_start + (current - loop_start + off) % loop_range;
    const auto slot = frame_index % _config.buffer_capacity;

    {
      std::shared_lock lock(_mutex);
      if (_ring_index[slot] == frame_index && _ring[slot]) continue;
    }

    pool.detach_task([this, frame_index, slot, gen] {
      if (_generation.load(std::memory_order_relaxed) != gen) return;
      load_into_slot(frame_index, slot);
    });
  }
}

} // namespace pc::devices::ply