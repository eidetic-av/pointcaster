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

#include "PointreceiverMessageInCHOP.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <logger.h>
#include <string>
#include <utility>

namespace {

constexpr const char *k_par_host = "Host";
constexpr const char *k_par_pointcloud_port = "Pointcloudport";
constexpr const char *k_par_message_port = "Messageport";
constexpr const char *k_par_active = "Active";
constexpr const char *k_par_reconnect = "Reconnect";

constexpr const char *k_par_address = "Address";

inline float nanf_() { return std::numeric_limits<float>::quiet_NaN(); }

static std::string safe_string(const char *s, const char *fallback) {
  if (s && s[0] != '\0') return std::string(s);
  return std::string(fallback ? fallback : "");
}

} // namespace

extern "C" {

DLLEXPORT void FillCHOPPluginInfo(TD::CHOP_PluginInfo *info) {
  info->apiVersion = TD::CHOPCPlusPlusAPIVersion;

  TD::OP_CustomOPInfo &customInfo = info->customOPInfo;
  customInfo.opType->setString("Pointreceivermessage");
  customInfo.opLabel->setString("Pointcaster Msg In");
  customInfo.opIcon->setString("PRM");

  customInfo.authorName->setString("Matt Hughes");
  customInfo.authorEmail->setString("matt@pointcaster.net");

  customInfo.minInputs = 0;
  customInfo.maxInputs = 0;
}

DLLEXPORT TD::CHOP_CPlusPlusBase *
CreateCHOPInstance(const TD::OP_NodeInfo *info) {
  return new PointreceiverMessageInCHOP(info);
}

DLLEXPORT void DestroyCHOPInstance(TD::CHOP_CPlusPlusBase *instance) {
  delete static_cast<PointreceiverMessageInCHOP *>(instance);
}

} // extern "C"

PointreceiverMessageInCHOP::PointreceiverMessageInCHOP(
    const TD::OP_NodeInfo *info)
    : _node_info(info) {
  pc::enable_file_logging("PointreceiverMessageInCHOP");
  pc::logger()->trace("CTOR this={} node_info={}", static_cast<void *>(this),
                      static_cast<const void *>(info));
}

PointreceiverMessageInCHOP::~PointreceiverMessageInCHOP() {
  pc::logger()->trace("DTOR begin this={}", static_cast<void *>(this));
  _connection_handle.reset();
  _selected_value_cached.reset();
  pc::logger()->trace("DTOR end this={}", static_cast<void *>(this));
}

void PointreceiverMessageInCHOP::getGeneralInfo(
    TD::CHOP_GeneralInfo *general_info, const TD::OP_Inputs *, void *) {
  // generator
  general_info->cookEveryFrameIfAsked = true;
  general_info->timeslice = false;
}

void PointreceiverMessageInCHOP::setupParameters(
    TD::OP_ParameterManager *manager, void *) {
  // Connection page (match your POP style)
  {
    TD::OP_StringParameter sp;
    sp.name = k_par_host;
    sp.label = "Host";
    sp.page = "Connection";
    sp.defaultValue = "127.0.0.1";
    auto res = manager->appendString(sp);
    assert(res == TD::OP_ParAppendResult::Success);
  }
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
    auto res = manager->appendInt(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }
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
    auto res = manager->appendInt(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }
  {
    TD::OP_NumericParameter np;
    np.name = k_par_active;
    np.label = "Active";
    np.page = "Connection";
    np.defaultValues[0] = 1.0;
    auto res = manager->appendToggle(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }
  {
    TD::OP_NumericParameter np;
    np.name = k_par_reconnect;
    np.label = "Reconnect";
    np.page = "Connection";
    auto res = manager->appendPulse(np);
    assert(res == TD::OP_ParAppendResult::Success);
  }

  // Message page
  {
    TD::OP_StringParameter sp;
    sp.name = k_par_address;
    sp.label = "Address";
    sp.page = "Message";
    sp.defaultValue = "";
    auto res = manager->appendString(sp);
    assert(res == TD::OP_ParAppendResult::Success);
  }
}

PointreceiverMessageInCHOP::OperatorSettings
PointreceiverMessageInCHOP::readSettings(const TD::OP_Inputs *inputs) const {
  OperatorSettings s{};

  s.connection.host =
      safe_string(inputs->getParString(k_par_host), "127.0.0.1");
  s.connection.pointcloud_port =
      static_cast<std::uint16_t>(inputs->getParInt(k_par_pointcloud_port));
  s.connection.message_port =
      static_cast<std::uint16_t>(inputs->getParInt(k_par_message_port));

  s.active = (inputs->getParInt(k_par_active) != 0);

  s.address_id = safe_string(inputs->getParString(k_par_address), "");
  return s;
}

void PointreceiverMessageInCHOP::ensureConnection(
    const OperatorSettings &settings, bool reconnect_pulse) {
  const bool config_changed = (settings.connection != _cached_connection);
  const bool active_changed = (settings.active != _cached_active);

  _cached_connection = settings.connection;
  _cached_active = settings.active;

  if (!settings.active) {
    _connection_handle.reset();
    _selected_value_cached.reset();
    _selected_revision_cached = 0;
    _selected_id_cached.clear();
    return;
  }

  if (!_connection_handle || reconnect_pulse || config_changed ||
      active_changed) {
    _connection_handle =
        pr::td::PointreceiverConnectionRegistry::instance().acquire(
            settings.connection);
    _selected_value_cached.reset();
    _selected_revision_cached = 0;
    _selected_id_cached.clear();
  }
}

bool PointreceiverMessageInCHOP::peekSelectedValue(
    std::string_view id, pr::td::PointreceiverMessageValueView &out_view) {
  out_view.reset();
  if (!_connection_handle || id.empty()) return false;
  // last_seen_revision=0 => returns latest (if any)
  return _connection_handle->try_get_latest_message(id, 0, out_view);
}

void PointreceiverMessageInCHOP::setCachedValueFromView(
    const pr::td::PointreceiverMessageValueView &view) {
  _selected_value_cached.reset();
  _selected_value_cached = view; // copies shared_ptr pins
  _selected_revision_cached = view.revision;
  _selected_id_cached.assign(view.id.data(), view.id.size());
}

void PointreceiverMessageInCHOP::computeOutputShapeFromCached(
    int32_t &out_channels, int32_t &out_samples) const {
  out_channels = 0;
  out_samples = 1;

  _channel_names_cached.clear();

  const pr::td::PointreceiverMessageValue *v =
      _selected_value_cached.value ? _selected_value_cached.value : nullptr;
  if (!v) return;

  auto set_names = [&](std::initializer_list<const char *> names) {
    _channel_names_cached.clear();
    _channel_names_cached.reserve(names.size());
    for (const char *n : names) _channel_names_cached.emplace_back(n);
  };

  std::visit(
      [&](const auto &payload) {
        using T = std::decay_t<decltype(payload)>;

        // Scalars / signals
        if constexpr (std::is_same_v<T, std::monostate> ||
                      std::is_same_v<T, pr::td::PointreceiverMessageSignal> ||
                      std::is_same_v<T, float> || std::is_same_v<T, int> ||
                      std::is_same_v<T, pr::td::PointreceiverEndpointUpdate>) {
          out_channels = 1;
          out_samples = 1;
          set_names({"value"});
          return;
        }

        if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat2>) {
          out_channels = 2;
          out_samples = 1;
          set_names({"x", "y"});
          return;
        }

        if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat3>) {
          out_channels = 3;
          out_samples = 1;
          set_names({"x", "y", "z"});
          return;
        }

        if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat4>) {
          out_channels = 4;
          out_samples = 1;
          set_names({"x", "y", "z", "w"});
          return;
        }

        // Lists -> samples = element count
        if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat2List>) {
          out_channels = 2;
          out_samples = static_cast<int32_t>(payload.size());
          set_names({"x", "y"});
          return;
        }

        if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat3List>) {
          out_channels = 3;
          out_samples = static_cast<int32_t>(payload.size());
          set_names({"x", "y", "z"});
          return;
        }

        if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat4List>) {
          out_channels = 4;
          out_samples = static_cast<int32_t>(payload.size());
          set_names({"x", "y", "z", "w"});
          return;
        }

        if constexpr (std::is_same_v<T, pr::td::PointreceiverAabbList>) {
          out_channels = 6;
          out_samples = static_cast<int32_t>(payload.size());
          set_names({"minx", "miny", "minz", "maxx", "maxy", "maxz"});
          return;
        }

        if constexpr (std::is_same_v<T, pr::td::PointreceiverContoursList>) {
          // Flatten all contours; insert a NaN separator row between contours.
          std::size_t total_points = 0;
          if (!payload.empty()) {
            for (const auto &c : payload) total_points += c.size();
            total_points += (payload.size() - 1); // separators
          }
          out_channels = 2;
          out_samples = static_cast<int32_t>(total_points);
          set_names({"x", "y"});
          return;
        }

        // Fallback
        out_channels = 1;
        out_samples = 1;
        set_names({"value"});
      },
      *v);

  if (out_channels < 0) out_channels = 0;
  if (out_samples < 1) out_samples = 1;
}

bool PointreceiverMessageInCHOP::getOutputInfo(TD::CHOP_OutputInfo *info,
                                               const TD::OP_Inputs *inputs,
                                               void *) {
  if (!info) return false;

  const OperatorSettings settings = readSettings(inputs);
  const bool reconnect_pulse = (inputs->getParInt(k_par_reconnect) != 0);

  ensureConnection(settings, reconnect_pulse);

  _cached_address_id = settings.address_id;

  // If selection changed, clear cached revision so we accept latest
  // immediately.
  if (_selected_id_cached != _cached_address_id) {
    _selected_value_cached.reset();
    _selected_revision_cached = 0;
    _selected_id_cached = _cached_address_id;
  }

  if (_connection_handle && !_selected_id_cached.empty()) {
    pr::td::PointreceiverMessageValueView view{};
    const bool got = _connection_handle->try_get_latest_message(
        _selected_id_cached, _selected_revision_cached, view);
    if (got) {
      setCachedValueFromView(view);
    } else if (!_selected_value_cached.value) {
      pr::td::PointreceiverMessageValueView peek{};
      if (_connection_handle->try_get_latest_message(_selected_id_cached, 0,
                                                     peek)) {
        setCachedValueFromView(peek);
      }
    }
  }

  int32_t channels = 0;
  int32_t samples = 1;
  computeOutputShapeFromCached(channels, samples);

  _channels_cached = channels;
  _samples_cached = samples;

  info->numChannels = channels;
  info->numSamples = samples;
  info->startIndex = 0;
  return true;
}

void PointreceiverMessageInCHOP::getChannelName(int32_t index,
                                                TD::OP_String *name,
                                                const TD::OP_Inputs *, void *) {
  if (!name) return;

  if (index >= 0 &&
      index < static_cast<int32_t>(_channel_names_cached.size())) {
    name->setString(
        _channel_names_cached[static_cast<std::size_t>(index)].c_str());
    return;
  }

  const std::string n = std::format("chan{}", index);
  name->setString(n.c_str());
}

void PointreceiverMessageInCHOP::fillOutputFromCached(TD::CHOP_Output *output) {
  if (!output || output->numChannels <= 0 || output->numSamples <= 0) return;

  // Default initialise output to 0
  for (int c = 0; c < output->numChannels; ++c) {
    std::memset(output->channels[c], 0,
                static_cast<std::size_t>(output->numSamples) * sizeof(float));
  }

  const pr::td::PointreceiverMessageValue *v =
      _selected_value_cached.value ? _selected_value_cached.value : nullptr;
  if (!v) return;

  auto write_sample = [&](int chan, int samp, float value) {
    if (chan < 0 || chan >= output->numChannels) return;
    if (samp < 0 || samp >= output->numSamples) return;
    output->channels[chan][samp] = value;
  };

  std::visit(
      [&](const auto &payload) {
        using T = std::decay_t<decltype(payload)>;

        if constexpr (std::is_same_v<T, std::monostate>) {
          return;
        } else if constexpr (std::is_same_v<
                                 T, pr::td::PointreceiverMessageSignal>) {
          write_sample(0, 0, static_cast<float>(static_cast<int>(payload)));
        } else if constexpr (std::is_same_v<T, float>) {
          write_sample(0, 0, payload);
        } else if constexpr (std::is_same_v<T, int>) {
          write_sample(0, 0, static_cast<float>(payload));
        } else if constexpr (std::is_same_v<
                                 T, pr::td::PointreceiverEndpointUpdate>) {
          // pack as: active ? port : -port (simple)
          const float sign = payload.active ? 1.0f : -1.0f;
          write_sample(0, 0, sign * static_cast<float>(payload.port));
        } else if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat2>) {
          write_sample(0, 0, payload.x);
          write_sample(1, 0, payload.y);
        } else if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat3>) {
          write_sample(0, 0, payload.x);
          write_sample(1, 0, payload.y);
          write_sample(2, 0, payload.z);
        } else if constexpr (std::is_same_v<T, pr::td::PointreceiverFloat4>) {
          write_sample(0, 0, payload.x);
          write_sample(1, 0, payload.y);
          write_sample(2, 0, payload.z);
          write_sample(3, 0, payload.w);
        } else if constexpr (std::is_same_v<T,
                                            pr::td::PointreceiverFloat2List>) {
          const int n = std::min<int>(output->numSamples,
                                      static_cast<int>(payload.size()));
          for (int i = 0; i < n; ++i) {
            write_sample(0, i, payload[static_cast<std::size_t>(i)].x);
            write_sample(1, i, payload[static_cast<std::size_t>(i)].y);
          }
        } else if constexpr (std::is_same_v<T,
                                            pr::td::PointreceiverFloat3List>) {
          const int n = std::min<int>(output->numSamples,
                                      static_cast<int>(payload.size()));
          for (int i = 0; i < n; ++i) {
            const auto &e = payload[static_cast<std::size_t>(i)];
            write_sample(0, i, e.x);
            write_sample(1, i, e.y);
            write_sample(2, i, e.z);
          }
        } else if constexpr (std::is_same_v<T,
                                            pr::td::PointreceiverFloat4List>) {
          const int n = std::min<int>(output->numSamples,
                                      static_cast<int>(payload.size()));
          for (int i = 0; i < n; ++i) {
            const auto &e = payload[static_cast<std::size_t>(i)];
            write_sample(0, i, e.x);
            write_sample(1, i, e.y);
            write_sample(2, i, e.z);
            write_sample(3, i, e.w);
          }
        } else if constexpr (std::is_same_v<T, pr::td::PointreceiverAabbList>) {
          const int n = std::min<int>(output->numSamples,
                                      static_cast<int>(payload.size()));
          for (int i = 0; i < n; ++i) {
            const auto &a = payload[static_cast<std::size_t>(i)];
            write_sample(0, i, a.min[0]);
            write_sample(1, i, a.min[1]);
            write_sample(2, i, a.min[2]);
            write_sample(3, i, a.max[0]);
            write_sample(4, i, a.max[1]);
            write_sample(5, i, a.max[2]);
          }
        } else if constexpr (std::is_same_v<
                                 T, pr::td::PointreceiverContoursList>) {
          int write_i = 0;
          for (std::size_t ci = 0; ci < payload.size(); ++ci) {
            const auto &contour = payload[ci];
            for (const auto &p : contour) {
              if (write_i >= output->numSamples) return;
              write_sample(0, write_i, p.x);
              write_sample(1, write_i, p.y);
              ++write_i;
            }
            if (ci + 1 < payload.size()) {
              if (write_i >= output->numSamples) return;
              write_sample(0, write_i, nanf_());
              write_sample(1, write_i, nanf_());
              ++write_i;
            }
          }
        } else {
          write_sample(0, 0, 0.0f);
        }
      },
      *v);
}

void PointreceiverMessageInCHOP::execute(TD::CHOP_Output *output,
                                         const TD::OP_Inputs *inputs, void *) {
  ++_execute_count;

  const OperatorSettings settings = readSettings(inputs);
  const bool reconnect_pulse = (inputs->getParInt(k_par_reconnect) != 0);

  ensureConnection(settings, reconnect_pulse);
  _cached_address_id = settings.address_id;

  if (!_connection_handle || !_cached_active || _cached_address_id.empty()) {
    _selected_value_cached.reset();
    _selected_revision_cached = 0;
    _selected_id_cached = _cached_address_id;
    fillOutputFromCached(output);
    return;
  }

  // If selected id changed, reset revision so we accept the latest immediately.
  if (_selected_id_cached != _cached_address_id) {
    _selected_value_cached.reset();
    _selected_revision_cached = 0;
    _selected_id_cached = _cached_address_id;
  }

  // Pull newest revision (if any), otherwise keep last cached payload.
  pr::td::PointreceiverMessageValueView view{};
  const bool got = _connection_handle->try_get_latest_message(
      _selected_id_cached, _selected_revision_cached, view);

  if (got) { setCachedValueFromView(view); }

  fillOutputFromCached(output);

  // Snapshot ids occasionally (cheap enough; if it becomes heavy, throttle)
  _known_ids_cached = _connection_handle
                          ? _connection_handle->known_message_ids()
                          : std::vector<std::string>{};
}

int32_t PointreceiverMessageInCHOP::getNumInfoCHOPChans(void *) { return 6; }

void PointreceiverMessageInCHOP::getInfoCHOPChan(int32_t index,
                                                 TD::OP_InfoCHOPChan *chan,
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
    chan->name->setString("messagesReceived");
    chan->value = static_cast<float>(stats.messages_received);
  } else if (index == 2) {
    chan->name->setString("messageWorkerLastMs");
    chan->value = stats.message_worker_last_ms;
  } else if (index == 3) {
    chan->name->setString("messageWorkerAvgMs");
    chan->value = stats.message_worker_avg_ms;
  } else if (index == 4) {
    chan->name->setString("selectedRevision");
    chan->value = static_cast<float>(_selected_revision_cached);
  } else if (index == 5) {
    chan->name->setString("knownIdsCount");
    chan->value = static_cast<float>(_known_ids_cached.size());
  }
}

bool PointreceiverMessageInCHOP::getInfoDATSize(TD::OP_InfoDATSize *info_size,
                                                void *) {
  if (!info_size) return false;

  if (_connection_handle) {
    _known_ids_cached = _connection_handle->known_message_ids();
  } else {
    _known_ids_cached.clear();
  }

  // Rows: header + each id
  info_size->rows = static_cast<int32_t>(_known_ids_cached.size()) + 1;
  info_size->cols = 2;
  info_size->byColumn = false;
  return true;
}

void PointreceiverMessageInCHOP::getInfoDATEntries(
    int32_t index, int32_t, TD::OP_InfoDATEntries *entries, void *) {
  if (!entries || !entries->values || !entries->values[0] ||
      !entries->values[1])
    return;

  auto set_row = [&](const char *k, const char *v) {
    entries->values[0]->setString(k ? k : "");
    entries->values[1]->setString(v ? v : "");
  };

  if (index == 0) {
    set_row("known_message_ids", "copy/paste into Address parameter");
    return;
  }

  const int32_t i = index - 1;
  if (i < 0 || i >= static_cast<int32_t>(_known_ids_cached.size())) {
    set_row("", "");
    return;
  }

  const std::string &id = _known_ids_cached[static_cast<std::size_t>(i)];
  set_row(std::to_string(i).c_str(), id.c_str());
}
