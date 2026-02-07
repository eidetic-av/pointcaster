#include "PointreceiverPlugin.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <format>
#include <logger.h>
#include <mutex>
#include <pointclouds.h>
#include <pointreceiver.h>
#include <thread>
#include <unordered_map>
#include <utility>

namespace pr::td {

namespace {

inline void hash_combine(std::size_t &seed, std::size_t v) noexcept {
  seed ^= v + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

constexpr std::uint32_t clamp_u32(std::uint32_t v, std::uint32_t lo,
                                  std::uint32_t hi) noexcept {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

} // namespace

PointreceiverGlobalConfig &global_config() noexcept {
  static PointreceiverGlobalConfig cfg{};
  return cfg;
}

std::size_t PointreceiverConnectionConfigHash::operator()(
    const PointreceiverConnectionConfig &k) const noexcept {
  std::size_t seed = 0;
  hash_combine(seed, std::hash<std::string>{}(k.host));
  hash_combine(seed, std::hash<std::uint16_t>{}(k.pointcloud_port));
  hash_combine(seed, std::hash<std::uint16_t>{}(k.message_port));
  return seed;
}

struct ConnectionState {
  explicit ConnectionState(PointreceiverConnectionConfig cfg,
                           std::uint64_t connection_id)
      : config(std::move(cfg)), connection_id(connection_id) {

    // Read globals once at construction time.
    // These apply to this connection until it is destroyed/recreated.
    // TODO maybe theres a way to have them hot reload
    auto &g = global_config();

    const std::uint32_t desired_cap =
        g.max_points_capacity.load(std::memory_order_relaxed);
    const std::uint32_t desired_pool =
        g.frame_pool_size.load(std::memory_order_relaxed);

    // TODO max sizes... necessary?
    capacity_points = clamp_u32(desired_cap, 1'024u, 5'000'000u);
    frame_pool_size = clamp_u32(desired_pool, 2u, 64u);

    const std::size_t cap = static_cast<std::size_t>(capacity_points);

    frame_pool.resize(static_cast<std::size_t>(frame_pool_size));
    for (auto &frame : frame_pool) {
      frame = std::make_shared<PointreceiverPointCloudFrame>();
      frame->positions_xyz.resize(cap * 3u);
      frame->colours_rgba.resize(cap * 4u);
      frame->indices.resize(cap);
      frame->sequence = 0;
      frame->num_points = 0;
    }

    pointreceiver_ctx = pointreceiver_create_context();
    if (pointreceiver_ctx) {
      client_name = std::format("touchdesigner_pr_{}", connection_id);
      pointreceiver_set_client_name(pointreceiver_ctx, client_name.c_str());
      start_receivers();
      worker_thread =
          std::jthread([this](std::stop_token st) { worker_main(st); });
    }
  }

  ~ConnectionState() {
    if (worker_thread.joinable()) {
      worker_thread.request_stop();
      worker_thread.join();
    }

    stop_receivers();

    if (pointreceiver_ctx) {
      pointreceiver_destroy_context(pointreceiver_ctx);
      pointreceiver_ctx = nullptr;
    }

    // Drop published frame last.
    published_frame.store({}, std::memory_order_release);
  }

  void start_receivers() {
    if (!pointreceiver_ctx) return;

    const std::string pc_ep =
        std::format("tcp://{}:{}", config.host, config.pointcloud_port);
    const std::string msg_ep =
        std::format("tcp://{}:{}", config.host, config.message_port);

    pointreceiver_start_point_receiver(pointreceiver_ctx, pc_ep.c_str());
    pointreceiver_start_message_receiver(pointreceiver_ctx, msg_ep.c_str());
  }

  void stop_receivers() {
    if (!pointreceiver_ctx) return;
    pointreceiver_stop_message_receiver(pointreceiver_ctx);
    pointreceiver_stop_point_receiver(pointreceiver_ctx);
  }

  std::shared_ptr<PointreceiverPointCloudFrame> try_acquire_writable_frame() {
    // writer-only: select a pool entry that isn't currently published and isn't
    // held by any readers... use_count()==1 means only the pool holds it
    const std::size_t n = frame_pool.size();
    for (std::size_t attempt = 0; attempt < n; ++attempt) {
      const std::size_t idx = (frame_pool_cursor + attempt) % n;
      const auto &candidate = frame_pool[idx];
      if (candidate && candidate.use_count() == 1) {
        frame_pool_cursor = (idx + 1) % n;
        return candidate;
      }
    }
    return {};
  }

  void publish_frame(const std::shared_ptr<PointreceiverPointCloudFrame> &frame,
                     std::uint64_t seq, std::uint32_t num_points) noexcept {
    frame->sequence = seq;
    frame->num_points = num_points;

    auto as_const =
        std::static_pointer_cast<const PointreceiverPointCloudFrame>(frame);
    published_frame.store(as_const, std::memory_order_release);
    latest_sequence.store(seq, std::memory_order_release);
  }

  void worker_main(std::stop_token stop_token) {
    using clock = std::chrono::steady_clock;

    constexpr std::size_t window = 120;
    float ms_window[window]{};
    std::size_t ms_write = 0;
    std::size_t ms_count = 0;
    float ms_sum = 0.0f;

    while (!stop_token.stop_requested()) {
      if (!pointreceiver_ctx) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      pointreceiver_pointcloud_frame in_frame{};
      const bool got =
          pointreceiver_dequeue_point_cloud(pointreceiver_ctx, &in_frame, 10);
      if (!got) continue;

      const int n = in_frame.point_count;
      if (n <= 0 || !in_frame.positions || !in_frame.colours) continue;

      const auto t0 = clock::now();

      const std::uint32_t cap = capacity_points;
      const std::uint32_t in_points = static_cast<std::uint32_t>(n);
      const std::uint32_t num_points = (in_points <= cap) ? in_points : cap;

      auto out_frame = try_acquire_writable_frame();
      if (!out_frame) {
        // All frames are pinned by readers or currently published; drop this
        // frame.
        continue;
      }

      const auto *packed_pos =
          static_cast<const bob::types::position *>(in_frame.positions);
      const auto *packed_col =
          static_cast<const bob::types::color *>(in_frame.colours);

      float *pos_out = out_frame->positions_xyz.data();
      float *col_out = out_frame->colours_rgba.data();
      std::uint32_t *idx_out = out_frame->indices.data();

      for (std::uint32_t i = 0; i < num_points; ++i) {
        const auto p = packed_pos[i];
        pos_out[i * 3u + 0u] = static_cast<float>(p.x);
        pos_out[i * 3u + 1u] = static_cast<float>(p.y);
        pos_out[i * 3u + 2u] = static_cast<float>(p.z);

        const auto c = packed_col[i];
        col_out[i * 4u + 0u] = static_cast<float>(c.b) * (1.0f / 255.0f);
        col_out[i * 4u + 1u] = static_cast<float>(c.g) * (1.0f / 255.0f);
        col_out[i * 4u + 2u] = static_cast<float>(c.r) * (1.0f / 255.0f);
        col_out[i * 4u + 3u] = static_cast<float>(c.a) * (1.0f / 255.0f);

        idx_out[i] = i;
      }

      const std::uint64_t seq =
          frames_received.fetch_add(1, std::memory_order_relaxed) + 1ull;

      publish_frame(out_frame, seq, num_points);

      const float ms =
          std::chrono::duration<float, std::milli>(clock::now() - t0).count();

      const float oldest = ms_window[ms_write];
      if (ms_count < window) {
        ms_count += 1;
        ms_sum += ms;
      } else {
        ms_sum += ms - oldest;
      }
      ms_window[ms_write] = ms;
      ms_write = (ms_write + 1) % window;

      worker_last_ms_atomic.store(ms, std::memory_order_relaxed);
      worker_avg_ms_atomic.store(
          ms_count ? (ms_sum / static_cast<float>(ms_count)) : 0.0f,
          std::memory_order_relaxed);
    }
  }

  PointreceiverConnectionConfig config;
  std::uint64_t connection_id = 0;
  std::string client_name;

  pointreceiver_context *pointreceiver_ctx = nullptr;
  std::jthread worker_thread;

  std::uint32_t capacity_points = 500'000;
  std::uint32_t frame_pool_size = 4;

  std::vector<std::shared_ptr<PointreceiverPointCloudFrame>> frame_pool{};
  std::size_t frame_pool_cursor = 0;

  std::atomic<std::shared_ptr<const PointreceiverPointCloudFrame>>
      published_frame{};
  std::atomic<std::uint64_t> latest_sequence{0};
  std::atomic<std::uint64_t> frames_received{0};
  std::atomic<float> worker_last_ms_atomic{0.0f};
  std::atomic<float> worker_avg_ms_atomic{0.0f};
};

PointreceiverConnectionHandle::PointreceiverConnectionHandle(
    std::shared_ptr<ConnectionState> state)
    : _state(std::move(state)) {}

PointreceiverConnectionHandle::~PointreceiverConnectionHandle() = default;

const PointreceiverConnectionConfig &
PointreceiverConnectionHandle::config() const noexcept {
  return _state->config;
}

bool PointreceiverConnectionHandle::try_get_latest_frame(
    std::uint64_t last_seen_sequence,
    PointreceiverFrameView &out_view) const noexcept {
  auto frame = _state->published_frame.load(std::memory_order_acquire);
  if (!frame) return false;

  const std::uint64_t seq = frame->sequence;
  if (seq == 0 || seq == last_seen_sequence) return false;

  const std::uint32_t n = frame->num_points;
  if (n == 0) return false;

  out_view._frame = std::move(frame);
  out_view.sequence = seq;
  out_view.num_points = n;

  out_view.positions_xyz = std::span<const float>(
      out_view._frame->positions_xyz.data(), static_cast<std::size_t>(n) * 3u);
  out_view.colours_rgba = std::span<const float>(
      out_view._frame->colours_rgba.data(), static_cast<std::size_t>(n) * 4u);
  out_view.indices = std::span<const std::uint32_t>(
      out_view._frame->indices.data(), static_cast<std::size_t>(n));

  return true;
}

std::uint64_t PointreceiverConnectionHandle::latest_sequence() const noexcept {
  return _state->latest_sequence.load(std::memory_order_acquire);
}

PointreceiverConnectionHandle::Stats
PointreceiverConnectionHandle::stats() const noexcept {
  const ConnectionState &st = *_state;
  Stats s;
  s.frames_received = st.frames_received.load(std::memory_order_relaxed);
  s.worker_last_ms = st.worker_last_ms_atomic.load(std::memory_order_relaxed);
  s.worker_avg_ms = st.worker_avg_ms_atomic.load(std::memory_order_relaxed);
  return s;
}

struct PointreceiverConnectionRegistry::RegistryState {
  std::mutex mutex;

  std::unordered_map<PointreceiverConnectionConfig,
                     std::weak_ptr<ConnectionState>,
                     PointreceiverConnectionConfigHash>
      entries;

  std::atomic<std::uint64_t> next_connection_id{1};
};

PointreceiverConnectionRegistry &PointreceiverConnectionRegistry::instance() {
  static PointreceiverConnectionRegistry inst;
  return inst;
}

PointreceiverConnectionRegistry::PointreceiverConnectionRegistry()
    : _registry_state(std::make_unique<RegistryState>()) {}

PointreceiverConnectionRegistry::~PointreceiverConnectionRegistry() = default;

std::shared_ptr<PointreceiverConnectionHandle>
PointreceiverConnectionRegistry::acquire(
    const PointreceiverConnectionConfig &config) {
  RegistryState &rs = *_registry_state;

  std::lock_guard<std::mutex> lock(rs.mutex);

  if (auto it = rs.entries.find(config); it != rs.entries.end()) {
    if (auto existing_state = it->second.lock()) {
      return std::make_shared<PointreceiverConnectionHandle>(existing_state);
    }
    rs.entries.erase(it);
  }

  const std::uint64_t id =
      rs.next_connection_id.fetch_add(1, std::memory_order_relaxed);

  auto new_state = std::make_shared<ConnectionState>(config, id);
  rs.entries.emplace(config, new_state);

  return std::make_shared<PointreceiverConnectionHandle>(std::move(new_state));
}

} // namespace pr::td
