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

#include <pointreceiver.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

class PointreceiverPOP final : public TD::POP_CPlusPlusBase {
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
  struct ConnectionSettings {
    std::string host{"127.0.0.1"};
    std::uint16_t pointcloud_port{9992};
    std::uint16_t message_port{9002};
    bool active{true};

    bool operator==(const ConnectionSettings &other) const {
      return host == other.host && pointcloud_port == other.pointcloud_port &&
             message_port == other.message_port && active == other.active;
    }
    bool operator!=(const ConnectionSettings &other) const {
      return !(*this == other);
    }
  };

  ConnectionSettings readSettings(const TD::OP_Inputs *inputs) const;

  void requestReconfigure(ConnectionSettings new_settings);

  // pointreceiver C API lifetime
  void ensureContextCreated();

  // Worker thread lifecycle
  void ensureWorkerRunning();
  void stopWorker();
  void workerMain(std::stop_token stop_token);

  // Worker thread only
  bool initialisePointreceiver(const ConnectionSettings &settings);

  // TD thread only
  void ensureMainThreadBuffersAllocated();
  void outputLatestFrame(TD::POP_Output *output);

private:
  const TD::OP_NodeInfo *const _node_info = nullptr;
  TD::POP_Context *const _context = nullptr;

  int32_t _execute_count = 0;
  std::atomic<std::uint64_t> _frames_received{0};
  std::uint64_t _last_consumed_sequence = 0;
  std::uint32_t _last_published_num_points = 0;

  std::atomic<bool> _reconfigure_requested{false};
  std::atomic<std::shared_ptr<const ConnectionSettings>> _pending_settings;
  ConnectionSettings _cached_settings{};

  std::atomic<float> _worker_frame_last_ms{0.0f};
  std::atomic<float> _worker_frame_avg_ms{0.0f};

  pointreceiver_context *_pointreceiver_context = nullptr;

  std::jthread _worker_thread;

  struct SharedState;
  std::unique_ptr<SharedState> _state;
};
