#include "radio.h"
#include "log.h"
#include <chrono>
#include <map>
#include <numeric>
#include <zmq.hpp>
#include <fmt/format.h>
#include "./devices/device.h"
#include "snapshots.h"
#include <imgui.h>

namespace bob::pointcaster {

static std::mutex stats_access;
static std::vector<uint> send_durations;
static std::vector<float> frame_sizes;

uint stats_frame_counter = 0;
float avg_duration = 0;
std::pair<uint, uint> minmax_duration;
float avg_size = 0;
std::pair<float, float> minmax_size;

Radio::Radio() {
  _radio_thread = make_thread(_current_config);
}

 std::unique_ptr<std::jthread>
 Radio::make_thread(const RadioConfiguration config) {
   return std::make_unique<std::jthread>([config](std::stop_token st) {
	 using namespace std::chrono;
         using namespace std::chrono_literals;

         constexpr auto broadcast_rate = 33ms;

         zmq::context_t zmq_context;
         zmq::socket_t radio(zmq_context, zmq::socket_type::radio);
         // prioritise the latest frame
         radio.set(zmq::sockopt::sndhwm, 1);
         // and don't keep excess frames in memory
         radio.set(zmq::sockopt::linger, 0);

         auto destination = fmt::format("tcp://*:{}", config.port);
         // auto destination = fmt::format("tcp://192.168.1.10:{}", port);
         radio.bind(destination);
	 bob::log.info("Radio broadcasting on port %d", config.port);

         time_point<system_clock> start_send_time;
         time_point<system_clock> end_send_time;

	 using delta_time = duration<uint, milliseconds>;
	 uint delta_ms = 0;

	 uint broadcast_snapshot_frame_count = 0;

         while (!st.stop_requested()) {

           // TODO broadcast_rate needs to by dynamic based on when we have a
           // new point cloud
	   auto sleep_time = broadcast_rate - milliseconds(delta_ms);
	   if (sleep_time.count() > 0)
	     std::this_thread::sleep_for(sleep_time);

           start_send_time = system_clock::now();
	 std::size_t packet_bytes = 0;

	 if (snapshots::frames.size() != broadcast_snapshot_frame_count) {

	   if (snapshots::frames.size() == 0) {
	     zmq::message_t snapshot_frame_msg(std::string_view("clear"));
	     snapshot_frame_msg.set_group("snapshots");
	     radio.send(snapshot_frame_msg, zmq::send_flags::none);
	     packet_bytes += snapshot_frame_msg.size();
	   } else {
	     auto synthesized_snapshot_frame = std::reduce(
		 snapshots::frames.begin(), snapshots::frames.end(),
		 types::PointCloud{},
		 [](auto a, auto b) -> types::PointCloud { return a + b; });
	     auto bytes =
		 synthesized_snapshot_frame.serialize(config.compress_frames);
	     zmq::message_t snapshot_frames_msg(bytes);
	     snapshot_frames_msg.set_group("snapshots");
	     radio.send(snapshot_frames_msg, zmq::send_flags::none);
	     packet_bytes += snapshot_frames_msg.size();
           }
           broadcast_snapshot_frame_count = snapshots::frames.size();
         }

           auto live_point_cloud = bob::sensors::synthesized_point_cloud();
           if (live_point_cloud.size() > 0) {
             auto bytes = live_point_cloud.serialize(config.compress_frames);
             zmq::message_t point_cloud_msg(bytes);
             point_cloud_msg.set_group("live");
             radio.send(point_cloud_msg, zmq::send_flags::none);
	     packet_bytes += point_cloud_msg.size();
           }

           end_send_time = system_clock::now();
	   delta_ms = duration_cast<milliseconds>(end_send_time - start_send_time).count();

           if (config.capture_stats) {
             std::lock_guard lock(stats_access);
             send_durations.push_back(delta_ms);
             frame_sizes.push_back(packet_bytes / (float)1024);
             if (send_durations.size() > 300) send_durations.clear();
             if (frame_sizes.size() > 300) frame_sizes.clear();
           }
         }

	 bob::log.info("Ending radio broadcast");
       });
 }

void Radio::set_config(const RadioConfiguration config) {
  if (_radio_thread != nullptr) {
    _radio_thread->request_stop();
    _radio_thread->join();
  }
  _radio_thread = make_thread(config);
  _current_config = config;
}

void Radio::draw_imgui_window() {
  ImGui::PushID("RadioBroadcast");
  ImGui::SetNextWindowPos({350.0f, 200.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({200.0f, 100.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);

  ImGui::Begin("Radio Broadcast", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);

  auto frame_config = _current_config;

  ImGui::Checkbox("Enable compression", &frame_config.compress_frames);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::Checkbox("Capture Stats", &frame_config.capture_stats);
  ImGui::Spacing();
  if (frame_config.capture_stats) {
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
	avg_size =
	    std::reduce(frame_sizes.begin(), frame_sizes.end()) /
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

  if (frame_config != _current_config) set_config(frame_config);

  ImGui::PopItemWidth();
  ImGui::End();
  ImGui::PopID();
}

} // namespace bob::pointcaster
