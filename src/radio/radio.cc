#include "radio.h"
#include "../devices/device.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../network_globals.h"
#include "../operators/operator_host_config.gen.h"
#include "../parameters.h"
#include "../profiling.h"
#include "../snapshots.h"
#include <charconv>
#include <chrono>
#include <fmt/format.h>
#include <imgui.h>
#include <map>
#include <numeric>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tracy/Tracy.hpp>
#include <zmq.hpp>
#include <zpp_bits.h>


namespace pc::radio {

using namespace pc::parameters;

static std::mutex stats_access;
static std::vector<unsigned int> send_durations;
static std::vector<float> frame_sizes;

unsigned int stats_frame_counter = 0;
float avg_duration = 0;
std::pair<unsigned int, unsigned int> minmax_duration;
float avg_size = 0;
std::pair<float, float> minmax_size;

Radio::Radio(RadioConfiguration &config,
             pc::operators::SessionOperatorHost &session_operator_host,
             pc::client_sync::SyncServer &sync_server)
    : _config(config), _session_operator_host(session_operator_host),
      _sync_server(sync_server),
      _radio_thread(std::make_unique<std::jthread>([this](std::stop_token st) {
        using namespace std::chrono;
        using namespace std::chrono_literals;

        zmq::socket_t live_socket(network_globals::get_zmq_context(),
                                  zmq::socket_type::radio);
        // prioritise the latest frame,
        // and don't keep excess frames in memory on socket close
        live_socket.set(zmq::sockopt::sndhwm, 1);
        live_socket.set(zmq::sockopt::linger, 0);
        live_socket.set(zmq::sockopt::conflate, 1);

        auto destination =
            fmt::format("tcp://{}:{}", _config.address, _config.port);
        pc::logger->info("Starting live_socket on '{}'", destination);
        try {
          live_socket.bind(destination);
        } catch (const zmq::error_t &e) {
          pc::logger->error(e.what());
          return;
        }
        pc::logger->info("live_socket started", destination);

        zmq::socket_t static_socket(network_globals::get_zmq_context(),
                                  zmq::socket_type::radio);
        // static_socket.set(zmq::sockopt::sndhwm, 1);
        static_socket.set(zmq::sockopt::linger, 0);
        auto static_destination =
            fmt::format("tcp://{}:{}", _config.address, _config.port + 1);
        pc::logger->info("Starting static_socket on '{}'", static_destination);
        try {
          static_socket.bind(static_destination);
        } catch (const zmq::error_t &e) {
          pc::logger->error(e.what());
          return;
        }
        pc::logger->info("static_socket started", destination);

        // we create individual sockets per static frame source...
        // this ensures the latest frame from each frame source
        // reaches the client, still with a high water mark of 1
        std::unordered_map<std::string, zmq::socket_t> static_sockets;

        // convenience func to publish all active static sockets to clients
        const auto publish_static_sockets = [&]() {
          for (auto &kvp : static_sockets) {
            const auto &socket = kvp.second;
            const std::string_view endpoint{
                socket.get(zmq::sockopt::last_endpoint)};
            auto colon_pos = endpoint.rfind(':');
            auto port_sv = endpoint.substr(colon_pos + 1);
            size_t port;
            std::from_chars(port_sv.data(), port_sv.data() + port_sv.size(),
                            port);
            _sync_server.publish_endpoint(
                {.id = kvp.first, .port = port, .active = true});
          }
        };

        _sync_server.on_client_connected.push_back(publish_static_sockets);

        using delta_time = duration<unsigned int, milliseconds>;
        milliseconds delta_ms;

        unsigned int broadcast_snapshot_frame_count = 0;

        constexpr auto broadcast_rate = 33ms;
        auto next_send_time = steady_clock::now() + broadcast_rate;

        while (!st.stop_requested()) {
          pc::profiling::ProfilingZone radio_zone("Radio");

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
            pc::profiling::ProfilingZone send_zone("Send");
            auto start_send_time = steady_clock::now();

            // TODO this vector should defs be removed
            // the synthesized_point_cloud api just requires a reference to a
            // list which we dont have at the moment
            std::vector<operators::OperatorHostConfiguration>
                fix_me_operator_list{_session_operator_host._config};
            static auto last_session_operators = _session_operator_host._config;
            bool updated_session_operators =
                last_session_operators != _session_operator_host._config;
            last_session_operators = _session_operator_host._config;

            const auto session_id = _session_operator_host.session_id;

            struct PendingMessage {
              std::optional<zmq::socket_ref> socket;
              zmq::message_t msg;
              std::optional<std::string> source_id;
            };
            tbb::concurrent_vector<PendingMessage> pending;

            tbb::task_group task_group;

            task_group.run([&]() {
              {
                pc::profiling::ProfilingZone live_cloud_zone(
                    "Live point cloud");
                types::PointCloud live_point_cloud;
                {
                  pc::profiling::ProfilingZone construct_zone(
                      "Construct cloud");
                  live_point_cloud = pc::devices::synthesized_point_cloud(
                      fix_me_operator_list, session_id);
                }
                _session_operator_host._config = fix_me_operator_list[0];
                if (live_point_cloud.size() > 0) {
                  pc::profiling::ProfilingZone serialize_zone("Serialize");
                  auto bytes =
                      live_point_cloud.serialize(_config.compress_frames);
                  zmq::message_t point_cloud_msg(bytes);
                  point_cloud_msg.set_group("live");
                  pending.push_back({.socket = live_socket,
                                     .msg = std::move(point_cloud_msg)});
                }
              }
            });

            task_group.run([&]() {
              auto static_point_clouds =
                  pc::devices::static_point_clouds(fix_me_operator_list,
                                                       session_id);

              pc::profiling::ProfilingZone static_zone(
                  "Static point clouds");

              tbb::parallel_for(
                  size_t(0), static_point_clouds.size(), [&](size_t i) {
                    auto &static_cloud = static_point_clouds[i];
                    bool needs_update = static_cloud.requires_update ||
                                        updated_session_operators;
                    if (!needs_update) return;

                    bool should_compress = _config.compress_frames;
                    if (static_cloud.disable_compression)
                      should_compress = false;

                    // TODO should operator application really happen in this
                    // radio class? most probably not...
                    types::PointCloud point_cloud;
                    {
                      pc::profiling::ProfilingZone operator_zone(
                          "Apply operators");
                      point_cloud = pc::operators::apply(
                          static_cloud.point_cloud.get(), fix_me_operator_list,
                          session_id);
                    }

                    types::StaticPointCloudMessage id_tagged_msg;
                    {
                      pc::profiling::ProfilingZone serialize_zone(
                          "Serialize pointcloud");
                      id_tagged_msg = {
                          .source_id = static_cloud.source_id,
                          .data = point_cloud.serialize(should_compress)};
                    }

                    auto [data, out] = zpp::bits::data_out();
                    std::errc serialize_result;
                    {
                      pc::profiling::ProfilingZone serialize_msg_zone(
                          "Serialize message");
                      serialize_result = out(id_tagged_msg);
                    }
                    if (zpp::bits::failure(serialize_result)) {
                      pc::logger->error(
                          "unable to serialize persisent cloud message");
                    } else {
                      {
                        pc::profiling::ProfilingZone serialize_msg_zone("Send");
                        zmq::message_t msg(data);
                        msg.set_group("static");
                        pending.push_back(
                            {.socket = static_socket, .msg = std::move(msg)});
                        // const auto source_id = static_cloud.source_id;
                        // if (static_sockets.contains(source_id)) {
                        //   pending.push_back(
                        //       {.socket = static_sockets[source_id],
                        //        .msg = std::move(msg)});
                        // } else {
                        //   pending.push_back(
                        //       {.msg = std::move(msg), .source_id = source_id});
                        // }
                      }
                      static_cloud.requires_update.get() = false;
                    }
                  });
            });

            task_group.wait();

            for (auto &[socket, msg, source_id] : pending) {

              // if (source_id.has_value() && !socket.has_value()) {
              //   // we create individual sockets per static frame source...
              //   // this ensures the latest frame from each frame source
              //   // reaches the client, still with a high water mark of 1
              //   static_sockets.emplace(
              //       source_id.value(),
              //       zmq::socket_t(network_globals::get_zmq_context(),
              //                     zmq::socket_type::radio));
              //   auto &new_socket = static_sockets[source_id.value()];
              //   // prioritise the latest frame,
              //   // and don't keep excess frames in memory on close
              //   new_socket.set(zmq::sockopt::sndhwm, 1);
              //   new_socket.set(zmq::sockopt::linger, 0);
              //   new_socket.set(zmq::sockopt::conflate, 1);

              //   auto destination = fmt::format("tcp://{}:*", _config.address);
              //   try {
              //     new_socket.bind(destination);
              //   } catch (const zmq::error_t &e) {
              //     pc::logger->error(e.what());
              //     return;
              //   }
              //   const std::string_view last_endpoint{
              //       new_socket.get(zmq::sockopt::last_endpoint)};
              //   pc::logger->info("Started static channel for '{}' on '{}'",
              //                    source_id.value(), last_endpoint);
              //   auto colon_pos = last_endpoint.rfind(':');
              //   auto port_sv = last_endpoint.substr(colon_pos + 1);
              //   size_t port;
              //   std::from_chars(port_sv.data(), port_sv.data() + port_sv.size(),
              //                   port);
              //   _sync_server.publish_endpoint(
              //       {.id = source_id.value(), .port = port, .active = true});

              //   socket = new_socket;
              // }

              {
                pc::profiling::ProfilingZone send_zone("Send");
                socket->send(msg, zmq::send_flags::none);
              }
            }
          }
        }

        pc::logger->info("Ended radio thread");
      })) {
  declare_parameters("workspace", "radio", _config);
}

void Radio::draw_imgui_window() {

  ImGui::SetNextWindowSize({250.0f, 400.0f}, ImGuiCond_FirstUseEver);

  ImGui::Begin("Radio", nullptr);
  pc::gui::draw_parameters("radio");
  ImGui::End();

  return;
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
