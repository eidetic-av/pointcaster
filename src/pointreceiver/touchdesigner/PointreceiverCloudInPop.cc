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

#include "PointreceiverCloudInPOP.h"

#include <algorithm>
#include <array>
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
constexpr const char *k_par_address = "Address";
constexpr const char *k_par_message_port = "Messageport";
constexpr const char *k_par_active = "Active";
constexpr const char *k_par_reconnect = "Reconnect";

constexpr const char *k_par_global_max_points_capacity = "Maxpointscapacity";
constexpr const char *k_par_global_frame_pool_size = "Framepoolsize";

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

struct PointreceiverCloudInPOP::SharedState {
  struct OutputBuffers {
    TD::OP_SmartRef<TD::POP_Buffer> position_attribute_buffer;
    TD::OP_SmartRef<TD::POP_Buffer> colour_attribute_buffer;
    TD::OP_SmartRef<TD::POP_Buffer> index_buffer;
    TD::OP_SmartRef<TD::POP_Buffer> point_info_buffer;
    TD::OP_SmartRef<TD::POP_Buffer> topo_info_buffer;

    void clear() {
      position_attribute_buffer.release();
      colour_attribute_buffer.release();
      index_buffer.release();
      point_info_buffer.release();
      topo_info_buffer.release();
    }

    bool handles_valid() const {
      return position_attribute_buffer && colour_attribute_buffer &&
             index_buffer && point_info_buffer && topo_info_buffer;
    }

    bool mapping_ok() const {
      if (!handles_valid()) return false;
      return position_attribute_buffer->getData(nullptr) != nullptr &&
             colour_attribute_buffer->getData(nullptr) != nullptr &&
             index_buffer->getData(nullptr) != nullptr;
    }
  };

  static constexpr std::size_t rotation_count = 2;
  std::array<OutputBuffers, rotation_count> rotation{};
  std::uint32_t capacity_points = 0;
  std::size_t write_cursor = 0;
  std::size_t last_published_index = 0;
};

extern "C" {

DLLEXPORT void FillPOPPluginInfo(TD::POP_PluginInfo *info) {
  if (!info->setAPIVersion(TD::POPCPlusPlusAPIVersion)) {
    return;
  }

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
  return new PointreceiverCloudInPOP(info, context);
}

DLLEXPORT void DestroyPOPInstance(TD::POP_CPlusPlusBase *instance) {
  delete static_cast<PointreceiverCloudInPOP *>(instance);
}

} // extern "C"

PointreceiverCloudInPOP::PointreceiverCloudInPOP(
    const TD::OP_NodeInfo *node_info, TD::POP_Context *context)
    : _node_info(node_info), _context(context),
      _state(std::make_unique<SharedState>()) {
  pc::enable_file_logging("PointreceiverCloudInPOP");
  pc::logger()->trace(
      "CTOR this={} node_info={} context={}", static_cast<void *>(this),
      static_cast<const void *>(node_info), static_cast<void *>(context));
}

PointreceiverCloudInPOP::~PointreceiverCloudInPOP() {
  pc::logger()->trace("DTOR begin this={}", static_cast<void *>(this));
  _channel_subscription.reset();
  _connection_handle.reset();
  _state.reset();
  pc::logger()->trace("DTOR end this={}", static_cast<void *>(this));
}

void PointreceiverCloudInPOP::getGeneralInfo(TD::POP_GeneralInfo *general_info,
                                             const TD::OP_Inputs *, void *) {
  general_info->cookEveryFrameIfAsked = true;
}

void PointreceiverCloudInPOP::setupParameters(TD::OP_ParameterManager *manager,
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

  // Pointcloud address
  {
    TD::OP_StringParameter sp;
    sp.name = k_par_address;
    sp.label = "Address";
    sp.page = "Connection";
    sp.defaultValue = "";
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
    np.maxValues[0] = 5000000;
    np.clampMins[0] = true;
    np.clampMaxes[0] = true;

    np.minSliders[0] = 1024;
    np.maxSliders[0] = 5000000;

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

PointreceiverCloudInPOP::OperatorSettings
PointreceiverCloudInPOP::readSettings(const TD::OP_Inputs *inputs) const {
  OperatorSettings s{};

  if (const char *host_cstr = inputs->getParString(k_par_host);
      host_cstr && host_cstr[0] != '\0') {
    s.connection.host = host_cstr;
  } else {
    s.connection.host = "127.0.0.1";
  }

  if (const char *address_cstr = inputs->getParString(k_par_address);
      address_cstr && address_cstr[0] != '\0') {
    s.address = address_cstr;
  }

  s.connection.pointcloud_port =
      static_cast<std::uint16_t>(inputs->getParInt(k_par_pointcloud_port));
  s.connection.message_port =
      static_cast<std::uint16_t>(inputs->getParInt(k_par_message_port));

  s.active = (inputs->getParInt(k_par_active) != 0);

  return s;
}

void PointreceiverCloudInPOP::syncGlobalSettings(const TD::OP_Inputs *inputs) {
  auto &g = pr::td::global_config();

  const std::uint32_t requested_cap = static_cast<std::uint32_t>(
      inputs->getParInt(k_par_global_max_points_capacity));
  const std::uint32_t requested_pool = static_cast<std::uint32_t>(
      inputs->getParInt(k_par_global_frame_pool_size));

  g.max_points_capacity.store(requested_cap, std::memory_order_relaxed);
  g.frame_pool_size.store(requested_pool, std::memory_order_relaxed);
}

void PointreceiverCloudInPOP::ensureMainThreadBuffersAllocated(
    std::uint32_t required_capacity_points) {
  SharedState &st = *_state;

  const std::uint32_t cap =
      clamp_u32(required_capacity_points, k_min_points_capacity,
                k_max_points_capacity_hard);

  const bool all_slots_ok =
      st.capacity_points == cap &&
      std::ranges::all_of(st.rotation,
                          [](const auto &slot) { return slot.mapping_ok(); });
  if (all_slots_ok) return;

  const std::uint64_t pos_bytes =
      static_cast<std::uint64_t>(cap) * 3ull * sizeof(float);
  const std::uint64_t col_bytes =
      static_cast<std::uint64_t>(cap) * 4ull * sizeof(float);
  const std::uint64_t idx_bytes =
      static_cast<std::uint64_t>(cap) * sizeof(std::uint32_t);

  auto allocate_slot = [&](SharedState::OutputBuffers &slot) -> bool {
    slot.clear();

    slot.position_attribute_buffer =
        create_cpu_buffer(_context, TD::POP_BufferUsage::Attribute, pos_bytes,
                          TD::POP_BufferMode::SequentialWrite);
    slot.colour_attribute_buffer =
        create_cpu_buffer(_context, TD::POP_BufferUsage::Attribute, col_bytes,
                          TD::POP_BufferMode::SequentialWrite);
    slot.index_buffer =
        create_cpu_buffer(_context, TD::POP_BufferUsage::IndexBuffer, idx_bytes,
                          TD::POP_BufferMode::SequentialWrite);

    slot.point_info_buffer = create_cpu_buffer(
        _context, TD::POP_BufferUsage::PointInfoBuffer,
        sizeof(TD::POP_PointInfo), TD::POP_BufferMode::ReadWrite);

    slot.topo_info_buffer = create_cpu_buffer(
        _context, TD::POP_BufferUsage::TopologyInfoBuffer,
        sizeof(TD::POP_TopologyInfo), TD::POP_BufferMode::ReadWrite);

    return slot.mapping_ok();
  };

  for (auto &slot : st.rotation) {
    if (slot.mapping_ok() && st.capacity_points == cap) continue;
    if (!allocate_slot(slot)) {
      // one retry, mirroring the previous single-buffer best-effort behaviour
      (void)allocate_slot(slot);
    }
  }

  st.capacity_points = cap;
  st.write_cursor = 0;
  st.last_published_index = 0;
}

void PointreceiverCloudInPOP::outputLatestFrame(TD::POP_Output *output) {
  SharedState &st = *_state;

  // Allocate per current global capacity (keeps POP buffers consistent with
  // receiver cap)
  const auto &g = pr::td::global_config();
  const std::uint32_t desired_cap =
      g.max_points_capacity.load(std::memory_order_relaxed);
  ensureMainThreadBuffersAllocated(desired_cap);

  auto publish = [&](SharedState::OutputBuffers &buffers) {
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

      output->setAttribute(&buffers.position_attribute_buffer, pos_info,
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

      output->setAttribute(&buffers.colour_attribute_buffer, col_info, set_info,
                           nullptr);
    }

    // Index
    {
      TD::POP_IndexBufferInfo index_info;
      index_info.type = TD::POP_IndexType::UInt32;
      output->setIndexBuffer(&buffers.index_buffer, index_info, set_info,
                             nullptr);
    }

    // Info buffers (always publish, even when 0 points)
    {
      if (auto *point_info = static_cast<TD::POP_PointInfo *>(
              buffers.point_info_buffer->getData(nullptr))) {
        std::memset(point_info, 0, sizeof(TD::POP_PointInfo));
        point_info->numPoints = _last_published_num_points;
      }

      if (auto *topo_info = static_cast<TD::POP_TopologyInfo *>(
              buffers.topo_info_buffer->getData(nullptr))) {
        std::memset(topo_info, 0, sizeof(TD::POP_TopologyInfo));
        topo_info->pointPrimitivesStartIndex = 0;
        topo_info->pointPrimitivesCount = _last_published_num_points;
      }

      TD::POP_InfoBuffers info_bufs;
      info_bufs.pointInfo = buffers.point_info_buffer;
      info_bufs.topoInfo = buffers.topo_info_buffer;
      output->setInfoBuffers(&info_bufs, set_info, nullptr);
    }
  };

  SharedState::OutputBuffers &last_published =
      st.rotation[st.last_published_index];

  if (!last_published.handles_valid()) {
    // nothing we can publish
    return;
  }

  // Default: publish whatever our last known count is
  if (!_channel_subscription) {
    _last_published_num_points = 0;
    publish(last_published);
    return;
  }

  pr::td::PointreceiverFrameView view{};
  const bool got =
      _channel_subscription->try_get_latest_frame(_last_seen_sequence, view);

  if (!got || view.num_points == 0 || view.num_points > st.capacity_points) {
    // If no new frame, keep publishing last known
    publish(last_published);
    return;
  }

  // Write the fresh frame into the next slot
  const std::size_t write_index = st.write_cursor;
  st.write_cursor = (st.write_cursor + 1) % SharedState::rotation_count;
  SharedState::OutputBuffers &target = st.rotation[write_index];

  if (!target.mapping_ok()) {
    publish(last_published);
    return;
  }

  _last_seen_sequence = view.sequence;
  _last_published_num_points = view.num_points;
  _last_published_centroid = view.centroid_xyz;

  float *pos_out =
      static_cast<float *>(target.position_attribute_buffer->getData(nullptr));
  float *col_out =
      static_cast<float *>(target.colour_attribute_buffer->getData(nullptr));
  std::uint32_t *idx_out =
      static_cast<std::uint32_t *>(target.index_buffer->getData(nullptr));

  const std::size_t pos_count = static_cast<std::size_t>(view.num_points) * 3u;
  const std::size_t col_count = static_cast<std::size_t>(view.num_points) * 4u;
  const std::size_t idx_count = static_cast<std::size_t>(view.num_points);

  std::memcpy(pos_out, view.positions_xyz.data(), pos_count * sizeof(float));
  std::memcpy(col_out, view.colours_rgba.data(), col_count * sizeof(float));
  std::memcpy(idx_out, view.indices.data(), idx_count * sizeof(std::uint32_t));

  st.last_published_index = write_index;
  publish(target);
}

void PointreceiverCloudInPOP::execute(TD::POP_Output *output,
                                      const TD::OP_Inputs *inputs, void *) {
  ++_execute_count;
  syncGlobalSettings(inputs);

  const OperatorSettings settings = readSettings(inputs);
  const bool reconnect_pulse = inputs->getParInt(k_par_reconnect) != 0;
  const bool config_changed = settings.connection != _cached_connection;
  const bool active_changed = settings.active != _cached_active;
  const bool address_changed = settings.address != _cached_address;

  _cached_connection = settings.connection;
  _cached_active = settings.active;
  _cached_address = settings.address;

  if (!settings.active || settings.address.empty()) {
    _channel_subscription.reset();
    _connection_handle.reset();
    _last_seen_sequence = 0;
    _last_published_num_points = 0;
    outputLatestFrame(output);
    return;
  }

  if (!_connection_handle || config_changed || active_changed) {
    _connection_handle =
        pr::td::PointreceiverConnectionRegistry::instance().acquire(
            settings.connection);
    _channel_subscription.reset();
  }

  if (reconnect_pulse && _connection_handle) {
    _connection_handle->reconnect();
    _channel_subscription.reset();
    _last_seen_sequence = 0;
  }

  if (!_channel_subscription || address_changed) {
    _channel_subscription = _connection_handle->subscribe(settings.address);
    _last_seen_sequence = 0;
  }

  outputLatestFrame(output);
}

int32_t PointreceiverCloudInPOP::getNumInfoCHOPChans(void *) {
  return 8;
}

void PointreceiverCloudInPOP::getInfoCHOPChan(int32_t index,
                                              TD::OP_InfoCHOPChan *chan,
                                              void *) {
  if (!chan) return;

  if (index == 0) {
    chan->name->setString("execute_count");
    chan->value = static_cast<float>(_execute_count);
    return;
  }

  const auto stats = _connection_handle
                         ? _connection_handle->stats()
                         : pr::td::PointreceiverConnectionHandle::Stats{};

  if (index == 1) {
    chan->name->setString("frames_received");
    chan->value = static_cast<float>(stats.pointcloud_frames_received);
  } else if (index == 2) {
    chan->name->setString("last_sequence");
    chan->value = static_cast<float>(_last_seen_sequence);
  } else if (index == 3) {
    chan->name->setString("worker_frame_last_ms");
    chan->value = stats.pointcloud_worker_last_ms;
  } else if (index == 4) {
    chan->name->setString("worker_frame_avg_ms");
    chan->value = stats.pointcloud_worker_avg_ms;
  } else if (index == 5) {
    chan->name->setString("centroid_x");
    chan->value = _last_published_centroid[0];
  } else if (index == 6) {
    chan->name->setString("centroid_y");
    chan->value = _last_published_centroid[1];
  } else if (index == 7) {
    chan->name->setString("centroid_z");
    chan->value = _last_published_centroid[2];
  }
}

bool PointreceiverCloudInPOP::getInfoDATSize(TD::OP_InfoDATSize *info_size,
                                             void *) {
  if (!info_size) return false;
  info_size->rows = 9;
  info_size->cols = 2;
  info_size->byColumn = false;
  return true;
}

void PointreceiverCloudInPOP::getInfoDATEntries(int32_t index, int32_t,
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
    set_row("execute_count", std::to_string(_execute_count).c_str());
    break;
  case 1:
    set_row("frames_received",
            std::to_string(stats.pointcloud_frames_received).c_str());
    break;
  case 2:
    set_row("last_sequence", std::to_string(_last_seen_sequence).c_str());
    break;
  case 3:
    set_row("host", _cached_connection.host.c_str());
    break;
  case 4: {
    const std::string s = std::format("pointcloud={}, message={}",
                                      _cached_connection.pointcloud_port,
                                      _cached_connection.message_port);
    set_row("ports", s.c_str());
    break;
  }
  case 5:
    set_row("active", _cached_active ? "true" : "false");
    break;
  case 6:
    set_row("centroid_x", std::to_string(_last_published_centroid[0]).c_str());
    break;
  case 7:
    set_row("centroid_y", std::to_string(_last_published_centroid[1]).c_str());
    break;
  case 8:
    set_row("centroid_z", std::to_string(_last_published_centroid[2]).c_str());
    break;
  default:
    break;
  }
}
