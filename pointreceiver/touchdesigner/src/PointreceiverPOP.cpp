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
#include <cstdint>
#include <cstring>
#include <format>
#include <logger.h>
#include <string>
#include <windows.h>

namespace {

constexpr const char *k_par_host = "Host";
constexpr const char *k_par_pointcloud_port = "Pointcloudport";
constexpr const char *k_par_message_port = "Messageport";
constexpr const char *k_par_active = "Active";
constexpr const char *k_par_reconnect = "Reconnect";

constexpr const char *k_par_global_max_points_capacity = "Maxpointscapacity";
constexpr const char *k_par_global_frame_pool_size = "Framepoolsize";

// Fallback if user enters nonsense (also avoids allocating 0)
constexpr std::uint32_t k_min_points_capacity = 1024u;
constexpr std::uint32_t k_max_points_capacity_hard = 5'000'000u;

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

static std::uint32_t clamp_u32(std::uint32_t v, std::uint32_t lo,
                               std::uint32_t hi) noexcept {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

} // namespace

struct PointreceiverPOP::SharedState {
  struct OutputBuffers {
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

  OutputBuffers out{};
};

extern "C" {

DLLEXPORT void FillPOPPluginInfo(TD::POP_PluginInfo *info) {
  if (!info->setAPIVersion(TD::POPCPlusPlusAPIVersion)) { return; }

  info->customOPInfo.opType->setString("Pointreceiver");
  info->customOPInfo.opLabel->setString("Pointcaster Cloud In");
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

PointreceiverPOP::PointreceiverPOP(const TD::OP_NodeInfo *node_info,
                                   TD::POP_Context *context)
    : _node_info(node_info), _context(context),
      _state(std::make_unique<SharedState>()) {
  pc::enable_file_logging("PointreceiverPOP");
  pc::logger()->trace(
      "CTOR this={} node_info={} context={}", static_cast<void *>(this),
      static_cast<const void *>(node_info), static_cast<void *>(context));
}

PointreceiverPOP::~PointreceiverPOP() {
  pc::logger()->trace("DTOR begin this={}", static_cast<void *>(this));
  _connection_handle.reset();
  _state.reset();
  pc::logger()->trace("DTOR end this={}", static_cast<void *>(this));
}

void PointreceiverPOP::getGeneralInfo(TD::POP_GeneralInfo *general_info,
                                      const TD::OP_Inputs *, void *) {
  general_info->cookEveryFrameIfAsked = true;
}

void PointreceiverPOP::setupParameters(TD::OP_ParameterManager *manager,
                                       void *) {
  // Host
  {
    TD::OP_StringParameter sp;
    sp.name = k_par_host;
    sp.label = "Host";
    sp.page = "Connection";
    sp.defaultValue = "127.0.0.1";
    const auto res = manager->appendString(sp);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  // Pointcloud Port
  {
    TD::OP_NumericParameter np;
    np.name = k_par_pointcloud_port;
    np.label = "Pointcloud Port";
    np.page = "Connection";
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
    np.page = "Connection";
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
    np.page = "Connection";
    np.defaultValues[0] = 1.0;
    const auto res = manager->appendToggle(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  // Reconnect pulse
  {
    TD::OP_NumericParameter np;
    np.name = k_par_reconnect;
    np.label = "Reconnect";
    np.page = "Connection";
    const auto res = manager->appendPulse(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  // Global page
  {
    TD::OP_NumericParameter np;
    np.name = k_par_global_max_points_capacity;
    np.label = "Max Points Capacity";
    np.page = "Global";

    np.defaultValues[0] = 500000;

    np.minValues[0] = 1;
    np.maxValues[0] = 2000000;
    np.clampMins[0] = true;
    np.clampMaxes[0] = true;

    // slider range (fixes the "1 max" UI behaviour)
    np.minSliders[0] = 1024;
    np.maxSliders[0] = 2000000;

    const auto res = manager->appendInt(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  {
    TD::OP_NumericParameter np;
    np.name = k_par_global_frame_pool_size;
    np.label = "Frame Pool Size";
    np.page = "Global";

    np.defaultValues[0] = 4;

    np.minValues[0] = 2;
    np.maxValues[0] = 64;
    np.clampMins[0] = true;
    np.clampMaxes[0] = true;

    np.minSliders[0] = 2;
    np.maxSliders[0] = 64;

    const auto res = manager->appendInt(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }
}

PointreceiverPOP::OperatorSettings
PointreceiverPOP::readSettings(const TD::OP_Inputs *inputs) const {
  OperatorSettings s{};

  if (const char *host_cstr = inputs->getParString(k_par_host);
      host_cstr && host_cstr[0] != '\0') {
    s.connection.host = host_cstr;
  } else {
    s.connection.host = "127.0.0.1";
  }

  s.connection.pointcloud_port =
      static_cast<std::uint16_t>(inputs->getParInt(k_par_pointcloud_port));
  s.connection.message_port =
      static_cast<std::uint16_t>(inputs->getParInt(k_par_message_port));

  s.active = (inputs->getParInt(k_par_active) != 0);

  return s;
}

void PointreceiverPOP::syncGlobalSettings(const TD::OP_Inputs *inputs) {
  auto &g = pr::td::global_config();

  const std::uint32_t requested_cap = static_cast<std::uint32_t>(
      inputs->getParInt(k_par_global_max_points_capacity));
  const std::uint32_t requested_pool = static_cast<std::uint32_t>(
      inputs->getParInt(k_par_global_frame_pool_size));

  g.max_points_capacity.store(requested_cap, std::memory_order_relaxed);
  g.frame_pool_size.store(requested_pool, std::memory_order_relaxed);
}

void PointreceiverPOP::ensureMainThreadBuffersAllocated(
    std::uint32_t required_capacity_points) {
  SharedState &st = *_state;

  const std::uint32_t cap =
      clamp_u32(required_capacity_points, k_min_points_capacity,
                k_max_points_capacity_hard);

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
    void *pos = st.out.position_attribute_buffer->getData(nullptr);
    void *col = st.out.colour_attribute_buffer->getData(nullptr);
    void *idx = st.out.index_buffer->getData(nullptr);
    return (pos != nullptr) && (col != nullptr) && (idx != nullptr);
  };

  // Realloc if capacity changed or handles invalid
  if (st.out.handles_valid() && st.out.capacity_points == cap && mapping_ok())
    return;

  bool ok = allocate_handles() && mapping_ok();
  if (!ok) {
    st.out.clear();
    (void)(allocate_handles() && mapping_ok());
  }
}

void PointreceiverPOP::outputLatestFrame(TD::POP_Output *output) {
  SharedState &st = *_state;

  // Allocate per current global capacity (keeps POP buffers consistent with
  // receiver cap)
  const auto &g = pr::td::global_config();
  const std::uint32_t desired_cap =
      g.max_points_capacity.load(std::memory_order_relaxed);
  ensureMainThreadBuffersAllocated(desired_cap);

  if (!st.out.handles_valid()) {
    // nothing we can publish
    return;
  }

  auto publish_current = [&]() {
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

      output->setAttribute(&st.out.position_attribute_buffer, pos_info,
                           set_info, nullptr);
    }

    {
      TD::POP_AttributeInfo col_info;
      col_info.name = "Color";
      col_info.numComponents = 4;
      col_info.numColumns = 1;
      col_info.arraySize = 0;
      col_info.type = TD::POP_AttributeType::Float;
      col_info.qualifier = TD::POP_AttributeQualifier::Color;
      col_info.attribClass = TD::POP_AttributeClass::Point;

      output->setAttribute(&st.out.colour_attribute_buffer, col_info, set_info,
                           nullptr);
    }

    // Index
    {
      TD::POP_IndexBufferInfo index_info;
      index_info.type = TD::POP_IndexType::UInt32;
      output->setIndexBuffer(&st.out.index_buffer, index_info, set_info,
                             nullptr);
    }

    // Info buffers (always publish, even when 0 points)
    {
      if (auto *point_info = static_cast<TD::POP_PointInfo *>(
              st.out.point_info_buffer->getData(nullptr))) {
        std::memset(point_info, 0, sizeof(TD::POP_PointInfo));
        point_info->numPoints = _last_published_num_points;
      }

      if (auto *topo_info = static_cast<TD::POP_TopologyInfo *>(
              st.out.topo_info_buffer->getData(nullptr))) {
        std::memset(topo_info, 0, sizeof(TD::POP_TopologyInfo));
        topo_info->pointPrimitivesStartIndex = 0;
        topo_info->pointPrimitivesCount = _last_published_num_points;
      }

      TD::POP_InfoBuffers info_bufs;
      info_bufs.pointInfo = st.out.point_info_buffer;
      info_bufs.topoInfo = st.out.topo_info_buffer;
      output->setInfoBuffers(&info_bufs, set_info, nullptr);
    }
  };

  // Default: publish whatever our last known count is (including 0)
  if (!_connection_handle) {
    _last_published_num_points = 0;
    publish_current();
    return;
  }

  pr::td::PointreceiverFrameView view{};
  const bool got =
      _connection_handle->try_get_latest_frame(_last_seen_sequence, view);

  if (!got || view.num_points == 0 ||
      view.num_points > st.out.capacity_points) {
    // If no new frame, keep publishing last known (does not silently drop to 0)
    publish_current();
    return;
  }

  // Update state regardless of info-buffer mapping success
  _last_seen_sequence = view.sequence;
  _last_published_num_points = view.num_points;

  float *pos_out =
      static_cast<float *>(st.out.position_attribute_buffer->getData(nullptr));
  float *col_out =
      static_cast<float *>(st.out.colour_attribute_buffer->getData(nullptr));
  std::uint32_t *idx_out =
      static_cast<std::uint32_t *>(st.out.index_buffer->getData(nullptr));

  if (!pos_out || !col_out || !idx_out) {
    // Realloc once and retry mapping
    st.out.clear();
    ensureMainThreadBuffersAllocated(desired_cap);

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
      // Don’t force count to 0 here — just publish whatever we last had
      publish_current();
      return;
    }
  }

  const std::size_t pos_count = static_cast<std::size_t>(view.num_points) * 3u;
  const std::size_t col_count = static_cast<std::size_t>(view.num_points) * 4u;
  const std::size_t idx_count = static_cast<std::size_t>(view.num_points);

  std::memcpy(pos_out, view.positions_xyz.data(), pos_count * sizeof(float));
  std::memcpy(col_out, view.colours_rgba.data(), col_count * sizeof(float));
  std::memcpy(idx_out, view.indices.data(), idx_count * sizeof(std::uint32_t));

  publish_current();
}

void PointreceiverPOP::execute(TD::POP_Output *output,
                               const TD::OP_Inputs *inputs, void *) {
  ++_execute_count;

  syncGlobalSettings(inputs);

  const OperatorSettings settings = readSettings(inputs);
  const bool reconnect_pulse = (inputs->getParInt(k_par_reconnect) != 0);

  const bool config_changed = (settings.connection != _cached_connection);
  const bool active_changed = (settings.active != _cached_active);

  _cached_connection = settings.connection;
  _cached_active = settings.active;

  if (!settings.active) {
    _connection_handle.reset();
    _last_seen_sequence = 0;
    _last_published_num_points = 0;
    outputLatestFrame(output);
    return;
  }

  if (!_connection_handle || reconnect_pulse || config_changed ||
      active_changed) {
    _connection_handle =
        pr::td::PointreceiverConnectionRegistry::instance().acquire(
            settings.connection);
    _last_seen_sequence = 0;
    // keep last_published_num_points as-is; outputLatestFrame will refresh
  }

  outputLatestFrame(output);
}

int32_t PointreceiverPOP::getNumInfoCHOPChans(void *) { return 5; }

void PointreceiverPOP::getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan *chan,
                                       void *) {
  if (!chan) return;

  if (index == 0) {
    chan->name->setString("executeCount");
    chan->value = static_cast<float>(_execute_count);
    return;
  }

  const auto stats = _connection_handle
                         ? _connection_handle->stats()
                         : pr::td::PointreceiverConnectionHandle::Stats{};

  if (index == 1) {
    chan->name->setString("framesReceived");
    chan->value = static_cast<float>(stats.frames_received);
  } else if (index == 2) {
    chan->name->setString("lastSequence");
    chan->value = static_cast<float>(_last_seen_sequence);
  } else if (index == 3) {
    chan->name->setString("workerFrameLastMs");
    chan->value = stats.worker_last_ms;
  } else if (index == 4) {
    chan->name->setString("workerFrameAvgMs");
    chan->value = stats.worker_avg_ms;
  }
}

bool PointreceiverPOP::getInfoDATSize(TD::OP_InfoDATSize *info_size, void *) {
  if (!info_size) return false;
  info_size->rows = 6;
  info_size->cols = 2;
  info_size->byColumn = false;
  return true;
}

void PointreceiverPOP::getInfoDATEntries(int32_t index, int32_t,
                                         TD::OP_InfoDATEntries *entries,
                                         void *) {
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

  const auto stats = _connection_handle
                         ? _connection_handle->stats()
                         : pr::td::PointreceiverConnectionHandle::Stats{};

  switch (index) {
  case 0:
    set_row("executeCount", std::to_string(_execute_count).c_str());
    break;
  case 1:
    set_row("framesReceived", std::to_string(stats.frames_received).c_str());
    break;
  case 2:
    set_row("lastSequence", std::to_string(_last_seen_sequence).c_str());
    break;
  case 3: set_row("host", _cached_connection.host.c_str()); break;
  case 4: {
    const std::string s = std::format("pointcloud={}, message={}",
                                      _cached_connection.pointcloud_port,
                                      _cached_connection.message_port);
    set_row("ports", s.c_str());
    break;
  }
  case 5: set_row("active", _cached_active ? "true" : "false"); break;
  default: break;
  }
}
