#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace pr::td {

// forward-declared so registry + handle can share it without tags/friends
struct ConnectionState;

// identifies a unique connection to pointcaster that can be shared across POPs
// and CHOPs
struct PointreceiverConnectionConfig {
  std::string host;
  std::uint16_t pointcloud_port = 9992;
  std::uint16_t message_port = 9002;

  bool
  operator==(const PointreceiverConnectionConfig &) const noexcept = default;
};

struct PointreceiverConnectionConfigHash {
  std::size_t operator()(const PointreceiverConnectionConfig &k) const noexcept;
};

// Process-wide config shared across all instances.
// TODO this doesn't hot-reload, need to remove POPs and recreate to have these
// as new creation args
struct PointreceiverGlobalConfig {
  std::atomic<std::uint32_t> max_points_capacity{500'000};
  std::atomic<std::uint32_t> frame_pool_size{4};
};

// Accessor for process-wide global config.
PointreceiverGlobalConfig &global_config() noexcept;

// Backing storage for one published point-cloud frame.
// Reader code pins lifetime by holding a shared_ptr to this object.
struct PointreceiverPointCloudFrame {
  std::vector<float> positions_xyz;   // cap * 3
  std::vector<float> colours_rgba;    // cap * 4
  std::vector<std::uint32_t> indices; // cap

  std::uint64_t sequence = 0;
  std::uint32_t num_points = 0;
};

// Read-only view of the most recently published frame.
struct PointreceiverFrameView {
  std::uint64_t sequence = 0;
  std::uint32_t num_points = 0;

  std::span<const float> positions_xyz;   // xyzxyz...
  std::span<const float> colours_rgba;    // rgbargba...
  std::span<const std::uint32_t> indices; // 0..n-1 size = num_points

  void reset() noexcept {
    sequence = 0;
    num_points = 0;
    positions_xyz = {};
    colours_rgba = {};
    indices = {};
    _frame.reset();
  }

private:
  friend class PointreceiverConnectionHandle;
  std::shared_ptr<const PointreceiverPointCloudFrame> _frame;
};

class PointreceiverConnectionHandle
    : public std::enable_shared_from_this<PointreceiverConnectionHandle> {
public:
  explicit PointreceiverConnectionHandle(
      std::shared_ptr<ConnectionState> state);
  ~PointreceiverConnectionHandle();

  const PointreceiverConnectionConfig &config() const noexcept;

  bool try_get_latest_frame(std::uint64_t last_seen_sequence,
                            PointreceiverFrameView &out_view) const noexcept;

  std::uint64_t latest_sequence() const noexcept;

  struct Stats {
    std::uint64_t frames_received = 0;
    float worker_last_ms = 0.0f;
    float worker_avg_ms = 0.0f;
  };

  Stats stats() const noexcept;

private:
  std::shared_ptr<ConnectionState> _state;
};

class PointreceiverConnectionRegistry {
public:
  static PointreceiverConnectionRegistry &instance();

  // identical configs share the same underlying connection state
  std::shared_ptr<PointreceiverConnectionHandle>
  acquire(const PointreceiverConnectionConfig &config);

private:
  PointreceiverConnectionRegistry();
  ~PointreceiverConnectionRegistry();

  struct RegistryState;
  std::unique_ptr<RegistryState> _registry_state;
};

} // namespace pr::td
