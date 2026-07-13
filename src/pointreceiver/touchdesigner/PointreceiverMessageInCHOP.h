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

#pragma once

#include "CHOP_CPlusPlusBase.h"
#include "PointreceiverPlugin.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class PointreceiverMessageInCHOP : public TD::CHOP_CPlusPlusBase {
public:
  explicit PointreceiverMessageInCHOP(const TD::OP_NodeInfo *info);
  ~PointreceiverMessageInCHOP() override;

  void getGeneralInfo(TD::CHOP_GeneralInfo *general_info,
                      const TD::OP_Inputs *inputs, void *reserved) override;

  bool getOutputInfo(TD::CHOP_OutputInfo *info, const TD::OP_Inputs *inputs,
                     void *reserved) override;

  void getChannelName(int32_t index, TD::OP_String *name,
                      const TD::OP_Inputs *inputs, void *reserved) override;

  void execute(TD::CHOP_Output *output, const TD::OP_Inputs *inputs,
               void *reserved) override;

  void setupParameters(TD::OP_ParameterManager *manager,
                       void *reserved) override;

  int32_t getNumInfoCHOPChans(void *reserved) override;
  void getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan *chan,
                       void *reserved) override;

  bool getInfoDATSize(TD::OP_InfoDATSize *info_size, void *reserved) override;
  void getInfoDATEntries(int32_t index, int32_t n_entries,
                         TD::OP_InfoDATEntries *entries,
                         void *reserved) override;

private:
  struct OperatorSettings {
    pr::td::PointreceiverConnectionConfig connection;
    bool active = true;
    std::string address_id;
  };

  OperatorSettings readSettings(const TD::OP_Inputs *inputs) const;

  void ensureConnection(const OperatorSettings &settings, bool reconnect_pulse);

  // Peek current value for sizing decisions (does not advance revision state)
  bool peekSelectedValue(std::string_view id,
                         pr::td::PointreceiverMessageValueView &out_view);

  // Determine output shape and channel naming for current cached value
  void computeOutputShapeFromCached(int32_t &out_channels,
                                    int32_t &out_samples) const;

  void
  setCachedValueFromView(const pr::td::PointreceiverMessageValueView &view);

  void fillOutputFromCached(TD::CHOP_Output *output);

private:
  const TD::OP_NodeInfo *const _node_info = nullptr;

  int32_t _execute_count = 0;

  pr::td::PointreceiverConnectionConfig _cached_connection{};
  bool _cached_active = true;
  std::string _cached_address_id{};

  std::shared_ptr<pr::td::PointreceiverConnectionHandle> _connection_handle;

  // Cache of the latest value for the selected id (pinned shared_ptr).
  std::string _selected_id_cached{};
  std::uint64_t _selected_revision_cached = 0;
  pr::td::PointreceiverMessageValueView _selected_value_cached{};

  // Output shape + channel naming cache for current cook
  mutable std::vector<std::string> _channel_names_cached;
  mutable int32_t _channels_cached = 0;
  mutable int32_t _samples_cached = 1;

  // Snapshot for Info DAT
  mutable std::vector<std::string> _known_ids_cached;
};
