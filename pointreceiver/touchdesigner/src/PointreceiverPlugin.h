#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
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
// TODO: these are only read once per connection at construction time...
struct PointreceiverGlobalConfig {
  std::atomic<std::uint32_t> max_points_capacity{500'000};
  std::atomic<std::uint32_t> frame_pool_size{4};
};

// Accessor for process-wide global config.
PointreceiverGlobalConfig &global_config() noexcept;

// ---------------------------
// Point cloud streams
// ---------------------------

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

// ---------------------------
// Messages
// ---------------------------

struct PointreceiverFloat2 {
  float x = 0.0f;
  float y = 0.0f;
};

struct PointreceiverFloat3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct PointreceiverFloat4 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
};

struct PointreceiverAabb {
  float min[3]{0.0f, 0.0f, 0.0f};
  float max[3]{0.0f, 0.0f, 0.0f};
};

struct PointreceiverEndpointUpdate {
  std::size_t port = 0;
  bool active = false;
};

using PointreceiverFloat2List = std::vector<PointreceiverFloat2>;
using PointreceiverFloat3List = std::vector<PointreceiverFloat3>;
using PointreceiverFloat4List = std::vector<PointreceiverFloat4>;
using PointreceiverAabbList = std::vector<PointreceiverAabb>;
using PointreceiverContour = std::vector<PointreceiverFloat2>;
using PointreceiverContoursList = std::vector<PointreceiverContour>;

// “Signal-only” message types (Connected, Heartbeat, etc.) can be represented
// as a simple tag, while parameter updates carry a typed payload.
enum class PointreceiverMessageSignal {
  Unknown = 0,
  Connected,
  ClientHeartbeat,
  ClientHeartbeatResponse,
  ParameterRequest,
};

// Payload for parameter updates (and endpoint updates).
using PointreceiverMessageValue =
    std::variant<std::monostate,             // no payload / not applicable
                 PointreceiverMessageSignal, // signal-only message
                 float, int, PointreceiverFloat2, PointreceiverFloat3,
                 PointreceiverFloat4, PointreceiverFloat2List,
                 PointreceiverFloat3List, PointreceiverFloat4List,
                 PointreceiverAabbList, PointreceiverContoursList,
                 PointreceiverEndpointUpdate>;

struct PointreceiverMessageValueView {
  std::uint64_t revision = 0;
  std::string_view id; // valid while this view lives (it holds shared_ptr)
  const PointreceiverMessageValue *value = nullptr;

  void reset() noexcept {
    revision = 0;
    id = {};
    value = nullptr;
    _value.reset();
    _id_storage.reset();
  }

private:
  friend class PointreceiverConnectionHandle;
  // Pin lifetime of payload and id storage.
  std::shared_ptr<const PointreceiverMessageValue> _value;
  std::shared_ptr<const std::string> _id_storage;
};

// ---------------------------
// Connection handle + registry
// ---------------------------

class PointreceiverConnectionHandle
    : public std::enable_shared_from_this<PointreceiverConnectionHandle> {
public:
  explicit PointreceiverConnectionHandle(
      std::shared_ptr<ConnectionState> state);
  ~PointreceiverConnectionHandle();

  const PointreceiverConnectionConfig &config() const noexcept;

  // Pointclouds
  bool try_get_latest_frame(std::uint64_t last_seen_sequence,
                            PointreceiverFrameView &out_view) const noexcept;

  std::uint64_t latest_sequence() const noexcept;

  // Messages
  // - returns false if id not known or no newer revision since
  // last_seen_revision
  bool try_get_latest_message(std::string_view id,
                              std::uint64_t last_seen_revision,
                              PointreceiverMessageValueView &out_view) const;

  std::vector<std::string> known_message_ids() const;

  struct Stats {
    // Pointcloud worker stats
    std::uint64_t pointcloud_frames_received = 0;
    float pointcloud_worker_last_ms = 0.0f;
    float pointcloud_worker_avg_ms = 0.0f;

    // Message worker stats
    std::uint64_t messages_received = 0;
    float message_worker_last_ms = 0.0f;
    float message_worker_avg_ms = 0.0f;
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
