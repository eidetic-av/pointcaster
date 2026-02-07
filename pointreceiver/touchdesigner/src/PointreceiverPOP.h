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

#include "POP_CPlusPlusBase.h"
#include "PointreceiverPlugin.h"

#include <cstdint>
#include <memory>

class PointreceiverPOP : public TD::POP_CPlusPlusBase {
public:
  PointreceiverPOP(const TD::OP_NodeInfo *node_info, TD::POP_Context *context);
  ~PointreceiverPOP() override;

  void getGeneralInfo(TD::POP_GeneralInfo *general_info,
                      const TD::OP_Inputs *inputs, void *reserved) override;

  void execute(TD::POP_Output *output, const TD::OP_Inputs *inputs,
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
  };

  OperatorSettings readSettings(const TD::OP_Inputs *inputs) const;

  void syncGlobalSettings(const TD::OP_Inputs *inputs);
  void ensureMainThreadBuffersAllocated(std::uint32_t required_capacity_points);
  void outputLatestFrame(TD::POP_Output *output);

private:
  const TD::OP_NodeInfo *const _node_info = nullptr;
  TD::POP_Context *const _context = nullptr;

  int32_t _execute_count = 0;

  pr::td::PointreceiverConnectionConfig _cached_connection{};
  bool _cached_active = true;

  std::shared_ptr<pr::td::PointreceiverConnectionHandle> _connection_handle;

  std::uint64_t _last_seen_sequence = 0;
  std::uint32_t _last_published_num_points = 0;

  struct SharedState;
  std::unique_ptr<SharedState> _state;
};
