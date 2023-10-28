#include "radio.h"
#include "../devices/device.h"
#include "../logger.h"
#include "../snapshots.h"
#include <Tracy/tracy/Tracy.hpp>
#include <chrono>
#include <fmt/format.h>
#include <imgui.h>
#include <map>
#include <numeric>
#include <zmq.hpp>

namespace pc::radio {

static std::mutex stats_access;
static std::vector<uint> send_durations;
static std::vector<float> frame_sizes;

uint stats_frame_counter = 0;
float avg_duration = 0;
std::pair<uint, uint> minmax_duration;
float avg_size = 0;
std::pair<float, float> minmax_size;

Radio::Radio(RadioConfiguration &config)
    : _config(config),
      _radio_thread(std::make_unique<std::jthread>([this](std::stop_token st) {
        using namespace std::chrono;
        using namespace std::chrono_literals;

        zmq::context_t zmq_context;
        zmq::socket_t radio(zmq_context, zmq::socket_type::radio);
        // prioritise the latest frame
        radio.set(zmq::sockopt::sndhwm, 1);
        // and don't keep excess frames in memory
        radio.set(zmq::sockopt::linger, 0);

        auto destination = fmt::format("tcp://*:{}", _config.port);
        // auto destination = fmt::format("tcp://192.168.1.10:{}", port);
        radio.bind(destination);
        pc::logger->info("Radio broadcasting on port {}", _config.port);

        using delta_time = duration<uint, milliseconds>;
        milliseconds delta_ms;

        uint broadcast_snapshot_frame_count = 0;

        constexpr auto broadcast_rate = 33ms;
        auto next_send_time = steady_clock::now() + broadcast_rate;

        while (!st.stop_requested()) {
          ZoneScopedN("Radio Tick");

          if (!_config.enabled) {
            std::this_thread::sleep_until(next_send_time);
            next_send_time += broadcast_rate;
            continue;
          }

          if (steady_clock::now() < next_send_time) {
            std::this_thread::sleep_until(next_send_time);
          }
          next_send_time += broadcast_rate;

          std::size_t packet_bytes = 0;
          {
            ZoneScopedN("Serialization and ZMQ Send");
            auto start_send_time = steady_clock::now();

            // if (snapshots::frames.size() != broadcast_snapshot_frame_count) {

            //   if (snapshots::frames.size() == 0) {
            //     zmq::message_t snapshot_frame_msg(std::string_view("clear"));
            //     snapshot_frame_msg.set_group("snapshots");
            //     radio.send(snapshot_frame_msg, zmq::send_flags::none);
            //     packet_bytes += snapshot_frame_msg.size();
            //   } else {
            //     auto synthesized_snapshot_frame = std::reduce(
            //         snapshots::frames.begin(), snapshots::frames.end(),
            //         types::PointCloud{},
            //         [](auto a, auto b) -> types::PointCloud { return a + b;
            //         });
            //     auto bytes = synthesized_snapshot_frame.serialize(
            //         _config.compress_frames);
            //     zmq::message_t snapshot_frames_msg(bytes);
            //     snapshot_frames_msg.set_group("snapshots");
            //     radio.send(snapshot_frames_msg, zmq::send_flags::none);
            //     packet_bytes += snapshot_frames_msg.size();
            //   }
            //   broadcast_snapshot_frame_count = snapshots::frames.size();
            // }

            auto live_point_cloud = pc::devices::synthesized_point_cloud();
            if (live_point_cloud.size() > 0) {
              auto bytes = live_point_cloud.serialize(_config.compress_frames);
              zmq::message_t point_cloud_msg(bytes);
              point_cloud_msg.set_group("live");
              radio.send(point_cloud_msg, zmq::send_flags::none);
              packet_bytes += point_cloud_msg.size();
            }

            delta_ms = duration_cast<milliseconds>(steady_clock::now() -
                                                   start_send_time);
          }

          if (_config.capture_stats) {
            std::lock_guard lock(stats_access);
            send_durations.push_back(delta_ms.count());
            frame_sizes.push_back(packet_bytes / (float)1024);
            constexpr auto max_stats_count = 300;
            if (send_durations.size() > max_stats_count) {
              send_durations.clear();
              frame_sizes.clear();
            }
          }
        }

        pc::logger->info("Ended radio thread");
      })) {}

void Radio::draw_imgui_window() {
  ImGui::PushID("Radio");
  ImGui::SetNextWindowPos({350.0f, 200.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({200.0f, 100.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);

  ImGui::Begin("Radio", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);

  ImGui::Checkbox("enabled", &_config.enabled);

  ImGui::Checkbox("compress_frames", &_config.compress_frames);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::Checkbox("Capture Stats", &_config.capture_stats);
  ImGui::Spacing();
  if (_config.capture_stats) {
    stats_frame_counter++;
    if (stats_frame_counter == 30) {
      stats_frame_counter = 0;
      std::lock_guard lock(stats_access);
      if (send_durations.size() > 0) {
        avg_duration =
            std::reduce(send_durations.begin(), send_durations.end()) /
            send_durations.size();
        const auto minmax =
            std::minmax_element(send_durations.begin(), send_durations.end());
        minmax_duration = {*minmax.first, *minmax.second};
      }
      if (frame_sizes.size() > 0) {
        avg_size = std::reduce(frame_sizes.begin(), frame_sizes.end()) /
                   frame_sizes.size();
        const auto minmax =
            std::minmax_element(frame_sizes.begin(), frame_sizes.end());
        minmax_size = {*minmax.first, *minmax.second};
      }
    }
    if (avg_duration != 0) {
      ImGui::Text("Packet Send Duration");
      ImGui::BeginTable("duration", 2);
      ImGui::TableNextColumn();
      ImGui::Text("Average");
      ImGui::TableNextColumn();
      ImGui::Text("%ims", (int)avg_duration);
      ImGui::TableNextColumn();
      ImGui::Text("Min");
      ImGui::TableNextColumn();
      ImGui::Text("%ims", minmax_duration.first);
      ImGui::TableNextColumn();
      ImGui::Text("Max");
      ImGui::TableNextColumn();
      ImGui::Text("%ims", minmax_duration.second);
      ImGui::EndTable();
      ImGui::Spacing();
      ImGui::Text("Packet Size");
      ImGui::BeginTable("duration", 2);
      ImGui::TableNextColumn();
      ImGui::Text("Average");
      ImGui::TableNextColumn();
      ImGui::Text("%fKB", avg_size);
      ImGui::TableNextColumn();
      ImGui::Text("Min");
      ImGui::TableNextColumn();
      ImGui::Text("%fKB", minmax_size.first);
      ImGui::TableNextColumn();
      ImGui::Text("Max");
      ImGui::TableNextColumn();
      ImGui::Text("%fKB", minmax_size.second);
      ImGui::EndTable();
      ImGui::Spacing();
      ImGui::Text("%.0f Mbps", (avg_size / (float)1024) * 8 * 30);
      ImGui::Spacing();
    } else {
      ImGui::Text("Calculating...");
    }
  }

  ImGui::PopItemWidth();
  ImGui::End();
  ImGui::PopID();
}

} // namespace pc::radio
