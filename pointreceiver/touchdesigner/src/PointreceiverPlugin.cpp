#include "PointreceiverPlugin.h"

#include <algorithm>
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

constexpr PointreceiverMessageSignal
to_signal(pointreceiver_message_type t) noexcept {
  switch (t) {
  case POINTRECEIVER_MSG_TYPE_CONNECTED:
    return PointreceiverMessageSignal::Connected;
  case POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT:
    return PointreceiverMessageSignal::ClientHeartbeat;
  case POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT_RESPONSE:
    return PointreceiverMessageSignal::ClientHeartbeatResponse;
  case POINTRECEIVER_MSG_TYPE_PARAMETER_REQUEST:
    return PointreceiverMessageSignal::ParameterRequest;
  default: return PointreceiverMessageSignal::Unknown;
  }
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
    auto &g = global_config();

    const std::uint32_t desired_cap =
        g.max_points_capacity.load(std::memory_order_relaxed);
    const std::uint32_t desired_pool =
        g.frame_pool_size.load(std::memory_order_relaxed);

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

      pointcloud_thread = std::jthread(
          [this](std::stop_token st) { pointcloud_worker_main(st); });

      message_thread =
          std::jthread([this](std::stop_token st) { message_worker_main(st); });
    }
  }

  ~ConnectionState() {
    if (pointcloud_thread.joinable()) {
      pointcloud_thread.request_stop();
      pointcloud_thread.join();
    }
    if (message_thread.joinable()) {
      message_thread.request_stop();
      message_thread.join();
    }

    stop_receivers();

    if (pointreceiver_ctx) {
      pointreceiver_destroy_context(pointreceiver_ctx);
      pointreceiver_ctx = nullptr;
    }

    // Drop published data last.
    published_frame.store({}, std::memory_order_release);
    latest_sequence.store(0, std::memory_order_release);

    // messages: clear under mutex
    {
      std::lock_guard<std::mutex> lock(messages_mutex);
      latest_messages.clear();
      known_message_ids.clear();
    }
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

  // ---------------------------
  // Pointcloud worker
  // ---------------------------

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

  void pointcloud_worker_main(std::stop_token stop_token) {
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
        // Keep your existing BGR->RGBA mapping.
        col_out[i * 4u + 0u] = static_cast<float>(c.b) * (1.0f / 255.0f);
        col_out[i * 4u + 1u] = static_cast<float>(c.g) * (1.0f / 255.0f);
        col_out[i * 4u + 2u] = static_cast<float>(c.r) * (1.0f / 255.0f);
        col_out[i * 4u + 3u] = static_cast<float>(c.a) * (1.0f / 255.0f);

        idx_out[i] = i;
      }

      const std::uint64_t seq =
          pointcloud_frames_received.fetch_add(1, std::memory_order_relaxed) +
          1ull;

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

      pointcloud_worker_last_ms_atomic.store(ms, std::memory_order_relaxed);
      pointcloud_worker_avg_ms_atomic.store(
          ms_count ? (ms_sum / static_cast<float>(ms_count)) : 0.0f,
          std::memory_order_relaxed);
    }
  }

  // ---------------------------
  // Message worker
  // ---------------------------

  struct MessageEntry {
    std::shared_ptr<const std::string> id_storage;
    std::shared_ptr<const PointreceiverMessageValue> value;
    std::uint64_t revision = 0;
  };

  static PointreceiverMessageValue
  convert_message(const pointreceiver_sync_message &msg) {
    // Signal-only types (no payload)
    if (msg.message_type != POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE &&
        msg.message_type != POINTRECEIVER_MSG_TYPE_ENDPOINT_UPDATE) {
      return PointreceiverMessageValue{to_signal(msg.message_type)};
    }

    if (msg.message_type == POINTRECEIVER_MSG_TYPE_ENDPOINT_UPDATE) {
      PointreceiverEndpointUpdate u;
      u.port = static_cast<std::size_t>(msg.value.endpoint_update_val.port);
      u.active = msg.value.endpoint_update_val.active;
      return PointreceiverMessageValue{u};
    }

    // Parameter update payload
    switch (msg.value_type) {
    case POINTRECEIVER_PARAM_VALUE_FLOAT:
      return PointreceiverMessageValue{msg.value.float_val};

    case POINTRECEIVER_PARAM_VALUE_INT:
      return PointreceiverMessageValue{msg.value.int_val};

    case POINTRECEIVER_PARAM_VALUE_FLOAT2: {
      PointreceiverFloat2 v;
      v.x = msg.value.float2_val.x;
      v.y = msg.value.float2_val.y;
      return PointreceiverMessageValue{v};
    }

    case POINTRECEIVER_PARAM_VALUE_FLOAT3: {
      PointreceiverFloat3 v;
      v.x = msg.value.float3_val.x;
      v.y = msg.value.float3_val.y;
      v.z = msg.value.float3_val.z;
      return PointreceiverMessageValue{v};
    }

    case POINTRECEIVER_PARAM_VALUE_FLOAT4: {
      PointreceiverFloat4 v;
      v.x = msg.value.float4_val.x;
      v.y = msg.value.float4_val.y;
      v.z = msg.value.float4_val.z;
      v.w = msg.value.float4_val.w;
      return PointreceiverMessageValue{v};
    }

    case POINTRECEIVER_PARAM_VALUE_FLOAT2LIST: {
      const auto &in = msg.value.float2_list_val;
      PointreceiverFloat2List out;
      out.reserve(in.count);
      for (std::size_t i = 0; i < in.count; ++i) {
        PointreceiverFloat2 v;
        v.x = in.data[i].x;
        v.y = in.data[i].y;
        out.push_back(v);
      }
      return PointreceiverMessageValue{std::move(out)};
    }

    case POINTRECEIVER_PARAM_VALUE_FLOAT3LIST: {
      const auto &in = msg.value.float3_list_val;
      PointreceiverFloat3List out;
      out.reserve(in.count);
      for (std::size_t i = 0; i < in.count; ++i) {
        PointreceiverFloat3 v;
        v.x = in.data[i].x;
        v.y = in.data[i].y;
        v.z = in.data[i].z;
        out.push_back(v);
      }
      return PointreceiverMessageValue{std::move(out)};
    }

    case POINTRECEIVER_PARAM_VALUE_FLOAT4LIST: {
      const auto &in = msg.value.float4_list_val;
      PointreceiverFloat4List out;
      out.reserve(in.count);
      for (std::size_t i = 0; i < in.count; ++i) {
        PointreceiverFloat4 v;
        v.x = in.data[i].x;
        v.y = in.data[i].y;
        v.z = in.data[i].z;
        v.w = in.data[i].w;
        out.push_back(v);
      }
      return PointreceiverMessageValue{std::move(out)};
    }

    case POINTRECEIVER_PARAM_VALUE_AABBLIST: {
      const auto &in = msg.value.aabb_list_val;
      PointreceiverAabbList out;
      out.reserve(in.count);
      for (std::size_t i = 0; i < in.count; ++i) {
        PointreceiverAabb a;
        for (int k = 0; k < 3; ++k) {
          a.min[k] = in.data[i].min[k];
          a.max[k] = in.data[i].max[k];
        }
        out.push_back(a);
      }
      return PointreceiverMessageValue{std::move(out)};
    }

    case POINTRECEIVER_PARAM_VALUE_CONTOURSLIST: {
      const auto &in = msg.value.contours_list_val;
      PointreceiverContoursList out;
      out.reserve(in.count);
      for (std::size_t ci = 0; ci < in.count; ++ci) {
        const auto &contour = in.data[ci];
        PointreceiverContour c;
        c.reserve(contour.count);
        for (std::size_t vi = 0; vi < contour.count; ++vi) {
          PointreceiverFloat2 v;
          v.x = contour.data[vi].x;
          v.y = contour.data[vi].y;
          c.push_back(v);
        }
        out.push_back(std::move(c));
      }
      return PointreceiverMessageValue{std::move(out)};
    }

    default: return PointreceiverMessageValue{std::monostate{}};
    }
  }

  void add_known_id(const std::string &id) {
    auto it = std::lower_bound(known_message_ids.begin(),
                               known_message_ids.end(), id);
    if (it == known_message_ids.end() || *it != id) {
      known_message_ids.insert(it, id);
    }
  }

  void message_worker_main(std::stop_token stop_token) {
    using clock = std::chrono::steady_clock;

    constexpr std::size_t window = 240;
    float ms_window[window]{};
    std::size_t ms_write = 0;
    std::size_t ms_count = 0;
    float ms_sum = 0.0f;

    while (!stop_token.stop_requested()) {
      if (!pointreceiver_ctx) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      const auto t0 = clock::now();

      pointreceiver_sync_message msg{};
      const bool got =
          pointreceiver_dequeue_message(pointreceiver_ctx, &msg, 10);
      if (!got) continue;

      // Copy id now (msg.id is a fixed buffer, but we want stable storage per
      // entry).
      const std::string id =
          (msg.id[0] != '\0') ? std::string(msg.id) : std::string{};

      const PointreceiverMessageValue converted = convert_message(msg);

      // Release any allocations inside msg (lists, contours, etc.)
      pointreceiver_free_sync_message(pointreceiver_ctx, &msg);

      // If there is no id (signal-only), we still count it but we don’t store
      // it keyed.
      if (!id.empty()) {
        auto id_storage = std::make_shared<std::string>(id);
        auto value_ptr = std::make_shared<PointreceiverMessageValue>(converted);

        {
          std::lock_guard<std::mutex> lock(messages_mutex);

          add_known_id(*id_storage);

          MessageEntry &entry = latest_messages[*id_storage];
          entry.id_storage = std::move(id_storage);
          entry.value = std::move(value_ptr);
          entry.revision = ++next_message_revision;
        }
      }

      messages_received.fetch_add(1, std::memory_order_relaxed);

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

      message_worker_last_ms_atomic.store(ms, std::memory_order_relaxed);
      message_worker_avg_ms_atomic.store(
          ms_count ? (ms_sum / static_cast<float>(ms_count)) : 0.0f,
          std::memory_order_relaxed);
    }
  }

  // ---------------------------
  // Data
  // ---------------------------

  PointreceiverConnectionConfig config;
  std::uint64_t connection_id = 0;
  std::string client_name;

  pointreceiver_context *pointreceiver_ctx = nullptr;

  // Two distinct worker threads.
  std::jthread pointcloud_thread;
  std::jthread message_thread;

  // Pointcloud config snapshot
  std::uint32_t capacity_points = 500'000;
  std::uint32_t frame_pool_size = 4;

  // Pointcloud storage
  std::vector<std::shared_ptr<PointreceiverPointCloudFrame>> frame_pool{};
  std::size_t frame_pool_cursor = 0;

  std::atomic<std::shared_ptr<const PointreceiverPointCloudFrame>>
      published_frame{};
  std::atomic<std::uint64_t> latest_sequence{0};

  // Pointcloud stats
  std::atomic<std::uint64_t> pointcloud_frames_received{0};
  std::atomic<float> pointcloud_worker_last_ms_atomic{0.0f};
  std::atomic<float> pointcloud_worker_avg_ms_atomic{0.0f};

  // Message storage
  mutable std::mutex messages_mutex;
  std::unordered_map<std::string, MessageEntry> latest_messages;
  std::vector<std::string> known_message_ids;

  std::uint64_t next_message_revision = 0;

  // Message stats
  std::atomic<std::uint64_t> messages_received{0};
  std::atomic<float> message_worker_last_ms_atomic{0.0f};
  std::atomic<float> message_worker_avg_ms_atomic{0.0f};
};

// ---------------------------
// Handle
// ---------------------------

PointreceiverConnectionHandle::PointreceiverConnectionHandle(
    std::shared_ptr<ConnectionState> state)
    : _state(std::move(state)) {}

PointreceiverConnectionHandle::~PointreceiverConnectionHandle() = default;

const PointreceiverConnectionConfig &
PointreceiverConnectionHandle::config() const noexcept {
  return _state->config;
}

// Pointcloud
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

// Messages
bool PointreceiverConnectionHandle::try_get_latest_message(
    std::string_view id, std::uint64_t last_seen_revision,
    PointreceiverMessageValueView &out_view) const {
  if (id.empty()) return false;

  ConnectionState &st = *_state;
  std::lock_guard<std::mutex> lock(st.messages_mutex);

  auto it = st.latest_messages.find(std::string(id));
  if (it == st.latest_messages.end()) return false;

  const ConnectionState::MessageEntry &entry = it->second;
  if (!entry.value || !entry.id_storage) return false;
  if (entry.revision == 0 || entry.revision == last_seen_revision) return false;

  out_view._value = entry.value;
  out_view._id_storage = entry.id_storage;
  out_view.revision = entry.revision;
  out_view.id = std::string_view(*out_view._id_storage);
  out_view.value = out_view._value.get();
  return true;
}

std::vector<std::string>
PointreceiverConnectionHandle::known_message_ids() const {
  ConnectionState &st = *_state;
  std::lock_guard<std::mutex> lock(st.messages_mutex);
  return st.known_message_ids; // already sorted in writer thread
}

PointreceiverConnectionHandle::Stats
PointreceiverConnectionHandle::stats() const noexcept {
  const ConnectionState &st = *_state;
  Stats s;
  s.pointcloud_frames_received =
      st.pointcloud_frames_received.load(std::memory_order_relaxed);
  s.pointcloud_worker_last_ms =
      st.pointcloud_worker_last_ms_atomic.load(std::memory_order_relaxed);
  s.pointcloud_worker_avg_ms =
      st.pointcloud_worker_avg_ms_atomic.load(std::memory_order_relaxed);

  s.messages_received = st.messages_received.load(std::memory_order_relaxed);
  s.message_worker_last_ms =
      st.message_worker_last_ms_atomic.load(std::memory_order_relaxed);
  s.message_worker_avg_ms =
      st.message_worker_avg_ms_atomic.load(std::memory_order_relaxed);
  return s;
}

// ---------------------------
// Registry
// ---------------------------

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
