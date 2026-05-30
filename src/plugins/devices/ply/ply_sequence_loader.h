#pragma once

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <pointcaster/point_cloud.h>
#include <shared_mutex>
#include <string>
#include <vector>

namespace pc::devices::ply {

// TODO
// Hardcoded layout from SessionRecorder::write_ply:
//   short x, short y, short z, uchar r, uchar g, uchar b
// this could probs be read in a type safe way using happly though??
constexpr size_t vertex_stride = 9;
constexpr size_t pos_offset = 0;
constexpr size_t color_offset = 6;

struct HeaderInfo {
  size_t data_offset;
  size_t vertex_count;
};

std::optional<HeaderInfo> scan_header(const char *data, size_t length);
std::shared_ptr<PointCloud> load_frame(const std::string &path);

class PlySequenceLoader {
public:
  struct Config {
    size_t buffer_capacity;
    size_t prefetch_ahead;
  };

  bool open(const std::filesystem::path &directory,
            const Config &config = {60, 30});

  std::shared_ptr<PointCloud> get_frame(size_t frame);

  size_t frame_count() const { return _file_paths.size(); }

  void invalidate();

private:
  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  Config _config;
  std::vector<std::string> _file_paths;

  std::vector<std::shared_ptr<PointCloud>> _ring;
  std::vector<size_t> _ring_index;
  mutable std::shared_mutex _mutex;

  std::atomic<size_t> _generation{0};

  std::shared_ptr<PointCloud> load_into_slot(size_t frame, size_t slot);
  void prefetch_from(size_t current);
};

} // namespace pc::devices::ply