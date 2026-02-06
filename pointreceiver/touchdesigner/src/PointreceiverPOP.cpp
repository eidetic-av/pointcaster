/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
 * and can only be used, and/or modified for use, in conjunction with
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement
 * (which also govern the use of this file). You may share or redistribute
 * a modified version of this file provided the following conditions are met:
 *
 * 1. The shared file or redistribution must retain the information set out
 * above and this list of conditions.
 * 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
 * to endorse or promote products derived from this file without specific
 * prior written permission from Derivative.
 */

#include "PointreceiverPOP.h"

#include <assert.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <windows.h>

#include <logger.h>
#include <pointclouds.h>

namespace {

constexpr const char *k_par_host = "Host";
constexpr const char *k_par_pointcloud_port = "Pointcloudport";
constexpr const char *k_par_message_port = "Messageport";
constexpr const char *k_par_active = "Active";
constexpr const char *k_par_reconnect = "Reconnect";

constexpr std::uint32_t k_log_every_n_frames = 60;

constexpr std::uint32_t k_max_points_capacity = 500'000;

inline std::string make_tcp_endpoint(std::string_view host,
                                     std::uint16_t port) {
  return std::format("tcp://{}:{}", host, port);
}

static TD::OP_SmartRef<TD::POP_Buffer>
create_cpu_buffer(TD::POP_Context *context, TD::POP_BufferUsage usage,
                  std::uint64_t size_bytes, TD::POP_BufferMode mode) {
  TD::POP_BufferInfo info;
  info.size = size_bytes;
  info.mode = mode;
  info.usage = usage;
  info.location = TD::POP_BufferLocation::CPU;
  info.stream = 0;
  return context->createBuffer(info, nullptr);
}

} // namespace

// -------------------------
// Shared state:
// - MAIN THREAD: allocates POP buffers (once) inside execute().
// - Worker thread: never calls createBuffer(), never resizes vectors.
// - Worker writes into pre-sized staging arrays.
// - Main thread memcpy staging -> POP buffer memory when publishing.
// -------------------------

struct PointreceiverPOP::SharedState final {
  struct OutputBuffers final {
    TD::OP_SmartRef<TD::POP_Buffer> position_attribute_buffer;
    TD::OP_SmartRef<TD::POP_Buffer> colour_attribute_buffer;
    TD::OP_SmartRef<TD::POP_Buffer> index_buffer;
    TD::OP_SmartRef<TD::POP_Buffer> point_info_buffer;
    TD::OP_SmartRef<TD::POP_Buffer> topo_info_buffer;

    std::uint32_t capacity_points = 0;

    void clear() {
      position_attribute_buffer.release();
      colour_attribute_buffer.release();
      index_buffer.release();
      point_info_buffer.release();
      topo_info_buffer.release();
      capacity_points = 0;
    }

    bool handles_valid() const {
      return capacity_points > 0 && position_attribute_buffer &&
             colour_attribute_buffer && index_buffer && point_info_buffer &&
             topo_info_buffer;
    }
  };

  struct StagingSlot final {
    std::vector<float> positions_xyz;   // cap * 3
    std::vector<float> colours_rgba;    // cap * 4
    std::vector<std::uint32_t> indices; // cap

    std::uint64_t sequence = 0;
    std::uint32_t num_points = 0;

    void reset_metadata() {
      sequence = 0;
      num_points = 0;
    }
  };

  OutputBuffers out{};

  StagingSlot slot_a{};
  StagingSlot slot_b{};

  std::atomic<StagingSlot *> front_ptr{&slot_a};
  StagingSlot *back_ptr = &slot_b;

  std::atomic<bool> has_new{false};
};

// -------------------------
// TD plugin entry points
// -------------------------

extern "C" {

DLLEXPORT void FillPOPPluginInfo(TD::POP_PluginInfo *info) {
  if (!info->setAPIVersion(TD::POPCPlusPlusAPIVersion)) { return; }

  info->customOPInfo.opType->setString("Pointreceiver");
  info->customOPInfo.opLabel->setString("Pointreceiver");
  info->customOPInfo.opIcon->setString("PRC");

  info->customOPInfo.authorName->setString("Matt Hughes");
  info->customOPInfo.authorEmail->setString("matt@pointcaster.net");

  info->customOPInfo.minInputs = 0;
  info->customOPInfo.maxInputs = 0;

  info->customOPInfo.opHelpURL->setString(
      "docs.pointcaster.net/Integrations/TouchDesigner");
}

DLLEXPORT TD::POP_CPlusPlusBase *CreatePOPInstance(const TD::OP_NodeInfo *info,
                                                   TD::POP_Context *context) {
  return new PointreceiverPOP(info, context);
}

DLLEXPORT void DestroyPOPInstance(TD::POP_CPlusPlusBase *instance) {
  delete static_cast<PointreceiverPOP *>(instance);
}

} // extern "C"

// -------------------------
// PointreceiverPOP
// -------------------------

PointreceiverPOP::PointreceiverPOP(const TD::OP_NodeInfo *node_info,
                                   TD::POP_Context *context)
    : _node_info(node_info), _context(context),
      _state(std::make_unique<SharedState>()) {
  _execute_count = 0;

  pc::enable_file_logging("PointreceiverPOP");

  // Make sure file sink exists for ctor logs too.

  pc::logger()->trace(
      "CTOR this={} node_info={} context={}", static_cast<void *>(this),
      static_cast<const void *>(node_info), static_cast<void *>(context));

  ensureContextCreated();

  // Only allocate *staging* here (main thread) - no TD buffer allocations here.
  {
    SharedState &st = *_state;
    const std::uint32_t cap = k_max_points_capacity;

    auto init_slot = [&](SharedState::StagingSlot &s) {
      s.positions_xyz.resize(static_cast<std::size_t>(cap) * 3u);
      s.colours_rgba.resize(static_cast<std::size_t>(cap) * 4u);
      s.indices.resize(static_cast<std::size_t>(cap));
      s.reset_metadata();
    };

    init_slot(st.slot_a);
    init_slot(st.slot_b);

    st.out.clear(); // buffers will be allocated in execute()
    pc::logger()->trace("CTOR: staging prealloc done cap={}", cap);
  }

  ensureWorkerRunning();
}

PointreceiverPOP::~PointreceiverPOP() {
  pc::logger()->trace("DTOR begin this={}", static_cast<void *>(this));
  stopWorker();
  pointreceiver_destroy_context(_pointreceiver_context);
  _state.reset();
  pc::logger()->trace("DTOR end this={}", static_cast<void *>(this));
}

void PointreceiverPOP::getGeneralInfo(TD::POP_GeneralInfo *general_info,
                                      const TD::OP_Inputs * /*inputs*/,
                                      void * /*reserved*/) {
  general_info->cookEveryFrameIfAsked = true;
}

void PointreceiverPOP::setupParameters(TD::OP_ParameterManager *manager,
                                       void * /*reserved*/) {
  pc::logger()->trace("setupParameters manager={}",
                      static_cast<void *>(manager));

  // Host
  {
    TD::OP_StringParameter sp;
    sp.name = k_par_host;
    sp.label = "Host";
    sp.defaultValue = "127.0.0.1";
    const auto res = manager->appendString(sp);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  // Pointcloud Port
  {
    TD::OP_NumericParameter np;
    np.name = k_par_pointcloud_port;
    np.label = "Pointcloud Port";
    np.defaultValues[0] = 9992;
    np.minSliders[0] = 1;
    np.maxSliders[0] = 65535;
    np.minValues[0] = 1;
    np.maxValues[0] = 65535;
    const auto res = manager->appendInt(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  // Message Port
  {
    TD::OP_NumericParameter np;
    np.name = k_par_message_port;
    np.label = "Message Port";
    np.defaultValues[0] = 9002;
    np.minSliders[0] = 1;
    np.maxSliders[0] = 65535;
    np.minValues[0] = 1;
    np.maxValues[0] = 65535;
    const auto res = manager->appendInt(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  // Active
  {
    TD::OP_NumericParameter np;
    np.name = k_par_active;
    np.label = "Active";
    np.defaultValues[0] = 1.0;
    const auto res = manager->appendToggle(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  // Reconnect pulse
  {
    TD::OP_NumericParameter np;
    np.name = k_par_reconnect;
    np.label = "Reconnect";
    const auto res = manager->appendPulse(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }
}

PointreceiverPOP::ConnectionSettings
PointreceiverPOP::readSettings(const TD::OP_Inputs *inputs) const {
  ConnectionSettings settings{};

  if (const char *host_cstr = inputs->getParString(k_par_host);
      host_cstr && host_cstr[0] != '\0') {
    settings.host = host_cstr;
  }

  settings.pointcloud_port =
      static_cast<std::uint16_t>(inputs->getParInt(k_par_pointcloud_port));
  settings.message_port =
      static_cast<std::uint16_t>(inputs->getParInt(k_par_message_port));
  settings.active = (inputs->getParInt(k_par_active) != 0);

  return settings;
}

void PointreceiverPOP::requestReconfigure(ConnectionSettings new_settings) {
  pc::logger()->trace(
      "requestReconfigure host='{}' pc_port={} msg_port={} active={}",
      new_settings.host, new_settings.pointcloud_port,
      new_settings.message_port, new_settings.active);

  _pending_settings.store(
      std::make_shared<const ConnectionSettings>(std::move(new_settings)),
      std::memory_order_release);

  _reconfigure_requested.store(true, std::memory_order_release);
}

void PointreceiverPOP::ensureContextCreated() {
  if (_pointreceiver_context) return;

  pc::logger()->trace("ensureContextCreated: creating pointreceiver context");
  _pointreceiver_context = pointreceiver_create_context();
  if (_pointreceiver_context) {
    pointreceiver_set_client_name(_pointreceiver_context, "touchdesigner");
    pc::logger()->info("context created {}",
                       static_cast<void *>(_pointreceiver_context));
  } else {
    pc::logger()->error("ensureContextCreated: FAILED to create context");
  }
}

void PointreceiverPOP::ensureWorkerRunning() {
  if (_worker_thread.joinable()) return;

  pc::logger()->info("starting worker thread");
  _worker_thread = std::jthread([this](std::stop_token st) { workerMain(st); });
}

void PointreceiverPOP::stopWorker() {
  if (_worker_thread.joinable()) {
    pc::logger()->trace("stopWorker: request_stop + join");
    _worker_thread.request_stop();
    _worker_thread.join();
  }
  pc::logger()->info("worker thread stopped");
}

bool PointreceiverPOP::initialisePointreceiver(
    const ConnectionSettings &settings) {
  if (!_pointreceiver_context) {
    pc::logger()->error("initialisePointreceiver: no context");
    return false;
  }

  pc::logger()->trace("initialisePointreceiver: begin active={}",
                      settings.active);

  pointreceiver_stop_message_receiver(_pointreceiver_context);
  pointreceiver_stop_point_receiver(_pointreceiver_context);

  if (!settings.active) {
    pc::logger()->trace(
        "initialisePointreceiver: inactive -> receivers stopped");
    return true;
  }

  const std::string pointcloud_endpoint =
      make_tcp_endpoint(settings.host, settings.pointcloud_port);
  const std::string message_endpoint =
      make_tcp_endpoint(settings.host, settings.message_port);

  pc::logger()->trace("initialisePointreceiver: pc='{}' msg='{}'",
                      pointcloud_endpoint, message_endpoint);

  const int pc_res = pointreceiver_start_point_receiver(
      _pointreceiver_context, pointcloud_endpoint.c_str());
  const int msg_res = pointreceiver_start_message_receiver(
      _pointreceiver_context, message_endpoint.c_str());

  pc::logger()->trace(
      "initialisePointreceiver: start results pc_res={} msg_res={}", pc_res,
      msg_res);
  return (pc_res == 0) && (msg_res == 0);
}

void PointreceiverPOP::workerMain(std::stop_token stop_token) {
  pc::logger()->trace("workerMain: enter this={} context={}",
                      static_cast<void *>(this),
                      static_cast<void *>(_pointreceiver_context));

  ConnectionSettings applied = _cached_settings;
  if (!initialisePointreceiver(applied)) {
    pc::logger()->error(
        "workerMain: unable to start pointreceiver (init failed)");
    return;
  }

  while (!stop_token.stop_requested()) {
    if (_reconfigure_requested.exchange(false, std::memory_order_acq_rel)) {
      if (auto pending = _pending_settings.load(std::memory_order_acquire)) {
        applied = *pending;
        pc::logger()->trace(
            "workerMain: applying reconfigure host='{}' pc_port={} "
            "msg_port={} active={}",
            applied.host, applied.pointcloud_port, applied.message_port,
            applied.active);
        const bool ok = initialisePointreceiver(applied);
        pc::logger()->trace("workerMain: reconfigure initialise ok={}", ok);
        _cached_settings = applied;
      }
    }

    if (!_pointreceiver_context || !applied.active) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    pointreceiver_pointcloud_frame in_frame{};
    const bool got =
        pointreceiver_dequeue_point_cloud(_pointreceiver_context, &in_frame, 5);
    if (!got) continue;

    const int n = in_frame.point_count;
    if (n <= 0 || !in_frame.positions || !in_frame.colours) {
      pc::logger()->trace(
          "workerMain: got frame but invalid n={} positions={} colours={}", n,
          static_cast<const void *>(in_frame.positions),
          static_cast<const void *>(in_frame.colours));
      continue;
    }

    SharedState &st = *_state;

    const std::uint32_t cap = k_max_points_capacity;
    const std::uint32_t num_points_in = static_cast<std::uint32_t>(n);
    const std::uint32_t num_points =
        (num_points_in <= cap) ? num_points_in : cap;

    const auto *packed_pos =
        static_cast<const bob::types::position *>(in_frame.positions);
    const auto *packed_col =
        static_cast<const bob::types::color *>(in_frame.colours);

    SharedState::StagingSlot &out = *st.back_ptr;

    float *pos_out = out.positions_xyz.data();
    float *col_out = out.colours_rgba.data();
    std::uint32_t *idx_out = out.indices.data();

    for (std::uint32_t i = 0; i < num_points; ++i) {
      const auto p = packed_pos[i];
      pos_out[i * 3u + 0u] = static_cast<float>(p.x);
      pos_out[i * 3u + 1u] = static_cast<float>(p.y);
      pos_out[i * 3u + 2u] = static_cast<float>(p.z);

      const auto c = packed_col[i];
      col_out[i * 4u + 0u] = static_cast<float>(c.r) * (1.0f / 255.0f);
      col_out[i * 4u + 1u] = static_cast<float>(c.g) * (1.0f / 255.0f);
      col_out[i * 4u + 2u] = static_cast<float>(c.b) * (1.0f / 255.0f);
      col_out[i * 4u + 3u] = static_cast<float>(c.a) * (1.0f / 255.0f);

      idx_out[i] = i;
    }

    out.num_points = num_points;
    const std::uint64_t seq =
        _frames_received.fetch_add(1, std::memory_order_relaxed) + 1ull;
    out.sequence = seq;

    if ((seq % k_log_every_n_frames) == 0u) {
      pc::logger()->trace(
          "workerMain: publish seq={} num_points={} (in={} cap={})", seq,
          num_points, num_points_in, cap);
    }

    SharedState::StagingSlot *previous_front =
        st.front_ptr.exchange(st.back_ptr, std::memory_order_acq_rel);
    st.back_ptr = previous_front;
    st.has_new.store(true, std::memory_order_release);
  }

  pc::logger()->trace("workerMain: exit");

  if (_pointreceiver_context) {
    pc::logger()->trace("workerMain: shutdown stop receivers");
    pointreceiver_stop_message_receiver(_pointreceiver_context);
    pointreceiver_stop_point_receiver(_pointreceiver_context);
  }
}

// Allocate (or re-allocate) POP buffers on the MAIN THREAD (inside execute()).
// Also, validate mapping by calling getData() once; if it returns nullptr,
// discard and re-create again. This specifically targets the failure you’re
// seeing.
void PointreceiverPOP::ensureMainThreadBuffersAllocated() {
  if (!_state) return;

  SharedState &st = *_state;
  const std::uint32_t cap = k_max_points_capacity;

  auto allocate_handles = [&]() -> bool {
    const std::uint64_t pos_bytes =
        static_cast<std::uint64_t>(cap) * 3ull * sizeof(float);
    const std::uint64_t col_bytes =
        static_cast<std::uint64_t>(cap) * 4ull * sizeof(float);
    const std::uint64_t idx_bytes =
        static_cast<std::uint64_t>(cap) * sizeof(std::uint32_t);

    st.out.clear();

    st.out.position_attribute_buffer =
        create_cpu_buffer(_context, TD::POP_BufferUsage::Attribute, pos_bytes,
                          TD::POP_BufferMode::SequentialWrite);
    st.out.colour_attribute_buffer =
        create_cpu_buffer(_context, TD::POP_BufferUsage::Attribute, col_bytes,
                          TD::POP_BufferMode::SequentialWrite);
    st.out.index_buffer =
        create_cpu_buffer(_context, TD::POP_BufferUsage::IndexBuffer, idx_bytes,
                          TD::POP_BufferMode::SequentialWrite);

    st.out.point_info_buffer = create_cpu_buffer(
        _context, TD::POP_BufferUsage::PointInfoBuffer,
        sizeof(TD::POP_PointInfo), TD::POP_BufferMode::ReadWrite);

    st.out.topo_info_buffer = create_cpu_buffer(
        _context, TD::POP_BufferUsage::TopologyInfoBuffer,
        sizeof(TD::POP_TopologyInfo), TD::POP_BufferMode::ReadWrite);

    st.out.capacity_points = cap;

    return st.out.handles_valid();
  };

  auto mapping_ok = [&]() -> bool {
    if (!st.out.handles_valid()) return false;

    // This is exactly what was failing for you:
    // if these are nullptr, treat buffers as not actually ready.
    void *pos = st.out.position_attribute_buffer->getData(nullptr);
    void *col = st.out.colour_attribute_buffer->getData(nullptr);
    void *idx = st.out.index_buffer->getData(nullptr);

    const bool ok = (pos != nullptr) && (col != nullptr) && (idx != nullptr);
    if (!ok) {
      pc::logger()->error(
          "ensureMainThreadBuffersAllocated: mapping test failed pos={} "
          "col={} idx={}",
          pos, col, idx);
    }
    return ok;
  };

  if (st.out.handles_valid() && mapping_ok()) { return; }

  pc::logger()->trace(
      "ensureMainThreadBuffersAllocated: allocating cap={} (main thread)", cap);

  // First attempt.
  bool ok = allocate_handles() && mapping_ok();

  // One retry: some TD internals may only be ready once execute() is active.
  if (!ok) {
    pc::logger()->warn(
        "ensureMainThreadBuffersAllocated: retry allocate after mapping "
        "failure");
    ok = allocate_handles() && mapping_ok();
  }

  pc::logger()->trace("ensureMainThreadBuffersAllocated: done ok={}", ok);
}

void PointreceiverPOP::outputLatestFrame(TD::POP_Output *output) {
  SharedState &st = *_state;

  ensureMainThreadBuffersAllocated();
  if (!st.out.handles_valid()) {
    pc::logger()->warn("outputLatestFrame: output buffers not allocated/valid");
    return;
  }

  auto publish_current = [&]() {
    if (_last_published_num_points == 0) return;

    TD::POP_SetBufferInfo set_info{};

    // P
    {
      TD::POP_AttributeInfo pos_info;
      pos_info.name = "P";
      pos_info.numComponents = 3;
      pos_info.numColumns = 1;
      pos_info.arraySize = 0;
      pos_info.type = TD::POP_AttributeType::Float;
      pos_info.qualifier = TD::POP_AttributeQualifier::None;
      pos_info.attribClass = TD::POP_AttributeClass::Point;

      auto tmp = st.out.position_attribute_buffer;
      output->setAttribute(&tmp, pos_info, set_info, nullptr);
    }

    // Color
    {
      TD::POP_AttributeInfo col_info;
      col_info.name = "Color";
      col_info.numComponents = 4;
      col_info.numColumns = 1;
      col_info.arraySize = 0;
      col_info.type = TD::POP_AttributeType::Float;
      col_info.qualifier = TD::POP_AttributeQualifier::Color;
      col_info.attribClass = TD::POP_AttributeClass::Point;

      auto tmp = st.out.colour_attribute_buffer;
      output->setAttribute(&tmp, col_info, set_info, nullptr);
    }

    // Index buffer
    {
      TD::POP_IndexBufferInfo index_info;
      index_info.type = TD::POP_IndexType::UInt32;

      auto tmp = st.out.index_buffer;
      output->setIndexBuffer(&tmp, index_info, set_info, nullptr);
    }

    // Info buffers
    {
      TD::POP_InfoBuffers info_bufs;
      info_bufs.pointInfo = st.out.point_info_buffer;
      info_bufs.topoInfo = st.out.topo_info_buffer;
      output->setInfoBuffers(&info_bufs, set_info, nullptr);
    }
  };

  const bool has_new = st.has_new.load(std::memory_order_acquire);
  SharedState::StagingSlot *front =
      st.front_ptr.load(std::memory_order_acquire);

  const bool frame_ready = has_new && front && front->sequence != 0 &&
                           front->sequence != _last_consumed_sequence;

  if (frame_ready) {
    const std::uint32_t num_points = front->num_points;

    if (num_points == 0 || num_points > st.out.capacity_points) {
      pc::logger()->warn("outputLatestFrame: invalid num_points={} cap={}",
                         num_points, st.out.capacity_points);
      publish_current();
      return;
    }

    float *pos_out = static_cast<float *>(
        st.out.position_attribute_buffer->getData(nullptr));
    float *col_out =
        static_cast<float *>(st.out.colour_attribute_buffer->getData(nullptr));
    std::uint32_t *idx_out =
        static_cast<std::uint32_t *>(st.out.index_buffer->getData(nullptr));

    if (!pos_out || !col_out || !idx_out) {
      // try realloc once (same as before)
      st.out.clear();
      ensureMainThreadBuffersAllocated();

      pos_out = st.out.position_attribute_buffer
                    ? static_cast<float *>(
                          st.out.position_attribute_buffer->getData(nullptr))
                    : nullptr;
      col_out = st.out.colour_attribute_buffer
                    ? static_cast<float *>(
                          st.out.colour_attribute_buffer->getData(nullptr))
                    : nullptr;
      idx_out = st.out.index_buffer ? static_cast<std::uint32_t *>(
                                          st.out.index_buffer->getData(nullptr))
                                    : nullptr;

      if (!pos_out || !col_out || !idx_out) {
        publish_current();
        return;
      }
    }

    // memcpy, update info buffers, update cached counts
    const std::size_t pos_count = static_cast<std::size_t>(num_points) * 3u;
    const std::size_t col_count = static_cast<std::size_t>(num_points) * 4u;
    const std::size_t idx_count = static_cast<std::size_t>(num_points);

    std::memcpy(pos_out, front->positions_xyz.data(),
                pos_count * sizeof(float));
    std::memcpy(col_out, front->colours_rgba.data(), col_count * sizeof(float));
    std::memcpy(idx_out, front->indices.data(),
                idx_count * sizeof(std::uint32_t));

    auto *point_info = static_cast<TD::POP_PointInfo *>(
        st.out.point_info_buffer->getData(nullptr));
    auto *topo_info = static_cast<TD::POP_TopologyInfo *>(
        st.out.topo_info_buffer->getData(nullptr));

    if (point_info && topo_info) {
      std::memset(point_info, 0, sizeof(TD::POP_PointInfo));
      std::memset(topo_info, 0, sizeof(TD::POP_TopologyInfo));
      point_info->numPoints = num_points;
      topo_info->pointPrimitivesStartIndex = 0;
      topo_info->pointPrimitivesCount = num_points;

      _last_published_num_points = num_points;
      _last_consumed_sequence = front->sequence;
      st.has_new.store(false, std::memory_order_release);
    }
  }

  publish_current();
}

void PointreceiverPOP::execute(TD::POP_Output *output,
                               const TD::OP_Inputs *inputs,
                               void * /*reserved*/) {
  ++_execute_count;

  ensureContextCreated();
  ensureWorkerRunning();

  ensureMainThreadBuffersAllocated();

  const ConnectionSettings new_settings = readSettings(inputs);
  const bool reconnect_pulse = (inputs->getParInt(k_par_reconnect) != 0);

  if (reconnect_pulse || (new_settings != _cached_settings)) {
    pc::logger()->trace(
        "execute: reconfigure requested reconnect_pulse={} host='{}' "
        "pc_port={} msg_port={} active={}",
        reconnect_pulse, new_settings.host, new_settings.pointcloud_port,
        new_settings.message_port, new_settings.active);
    _cached_settings = new_settings;
    requestReconfigure(new_settings);
  }

  outputLatestFrame(output);
}

int32_t PointreceiverPOP::getNumInfoCHOPChans(void * /*reserved*/) {
  return 3; // executeCount, framesReceived, lastSequence
}

void PointreceiverPOP::getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan *chan,
                                       void * /*reserved*/) {
  if (!chan) return;

  if (index == 0) {
    chan->name->setString("executeCount");
    chan->value = static_cast<float>(_execute_count);
  } else if (index == 1) {
    chan->name->setString("framesReceived");
    chan->value =
        static_cast<float>(_frames_received.load(std::memory_order_relaxed));
  } else if (index == 2) {
    chan->name->setString("lastSequence");
    chan->value = static_cast<float>(_last_consumed_sequence);
  }
}

bool PointreceiverPOP::getInfoDATSize(TD::OP_InfoDATSize *info_size,
                                      void * /*reserved*/) {
  if (!info_size) return false;

  info_size->rows = 6;
  info_size->cols = 2;
  info_size->byColumn = false;
  return true;
}

void PointreceiverPOP::getInfoDATEntries(int32_t index, int32_t /*n_entries*/,
                                         TD::OP_InfoDATEntries *entries,
                                         void * /*reserved*/) {
  if (!entries || !entries->values || !entries->values[0] ||
      !entries->values[1])
    return;

  char key[256]{};
  char val[1024]{};

  const auto set_row = [&](const char *k, const char *v) {
    strcpy_s(key, k);
    strcpy_s(val, v);
    entries->values[0]->setString(key);
    entries->values[1]->setString(val);
  };

  switch (index) {
  case 0:
    set_row("executeCount", std::to_string(_execute_count).c_str());
    break;
  case 1:
    set_row("framesReceived",
            std::to_string(_frames_received.load(std::memory_order_relaxed))
                .c_str());
    break;
  case 2:
    set_row("lastSequence", std::to_string(_last_consumed_sequence).c_str());
    break;
  case 3: set_row("host", _cached_settings.host.c_str()); break;
  case 4: {
    const std::string s = std::format("pointcloud={}, message={}",
                                      _cached_settings.pointcloud_port,
                                      _cached_settings.message_port);
    set_row("ports", s.c_str());
    break;
  }
  case 5: set_row("active", _cached_settings.active ? "true" : "false"); break;
  default: break;
  }
}
