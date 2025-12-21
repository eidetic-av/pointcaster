#include "k4a_body_tracking.h"
#include "../../structs.h"
#include <Corrade/PluginManager/AbstractManager.h>
#include <Eigen/Core>
#include <k4abt.hpp>
#include <k4abttypes.h>
#include <memory>

#include "k4abt_utils.h"

namespace pc::devices {

class AzureKinectBodyTracking final : public AbstractAzureKinectBodyTracking {
public:
  explicit AzureKinectBodyTracking(
      Corrade::PluginManager::AbstractManager &manager,
      Corrade::Containers::StringView plugin)
      : AbstractAzureKinectBodyTracking{manager, plugin} {}

  std::unique_ptr<k4abt::tracker> _tracker;

  using K4ASkeleton =
      std::array<std::pair<pc::types::position, pc::types::Float4>, K4ABT_JOINT_COUNT>;

  std::vector<K4ASkeleton> _skeletons{};

  void init(const k4a::calibration& calibration) override {

  }

  void enqueue_capture(const k4a::capture &capture) const override {}

  void track_bodies() override {
  //   using namespace Eigen;

  //   // reserve space for five skeletons
  //   _skeletons.reserve(5);

  //   while (!_stop_requested) {

  //     if (!_tracker || !_open || !_running) {
  //       std::this_thread::sleep_for(50ms);
  //       continue;
  //     }

  //     k4abt_frame_t body_frame_handle = nullptr;
  //     k4abt::frame body_frame(body_frame_handle);
  //     if (!_tracker->pop_result(&body_frame, 50ms)) continue;

  //     const auto body_count = body_frame.get_num_bodies();
  //     if (body_count == 0) continue;

  //     auto config = _last_config;
  //     // TODO make all the following serialize into the config
  //     auto alignment_center = Eigen::Vector3f(
  //         _alignment_center.x, _alignment_center.y, _alignment_center.z);
  //     auto aligned_position_offset = Eigen::Vector3f(
  //         _aligned_position_offset.x, _aligned_position_offset.y,
  //         _aligned_position_offset.z);

  //     // transform the skeletons based on device config
  //     // and place them into the _skeletons list
  //     _skeletons.clear();
  //     for (std::size_t body_num = 0; body_num < body_count; body_num++) {
  //       const k4abt_skeleton_t raw_skeleton = body_frame.get_body_skeleton(0);
  //       K4ASkeleton skeleton;
  //       // parse each joint
  //       for (std::size_t joint = 0; joint < K4ABT_JOINT_COUNT; joint++) {
  //         auto pos = raw_skeleton.joints[joint].position.xyz;
  //         auto orientation = raw_skeleton.joints[joint].orientation.wxyz;

  //         Vector3f pos_f(pos.x, pos.y, pos.z);
  //         Quaternionf ori_f(orientation.w, orientation.x, orientation.y,
  //                           orientation.z);

  //         // perform any auto-tilt
  //         {
  //           std::lock_guard lock(_auto_tilt_value_mutex);
  //           pos_f = _auto_tilt_value * pos_f;
  //           ori_f = _auto_tilt_value * ori_f;
  //         }
  //         // flip y and z axes for our world space
  //         pos_f = Vector3f(pos_f[0], -pos_f[1], -pos_f[2]);

  //         static constexpr auto rad = [](float deg) {
  //           constexpr auto mult = 3.141592654f / 180.0f;
  //           return deg * mult;
  //         };

  //         auto &transform = config.transform;

  //         // create the rotation around our center
  //         AngleAxisf rot_x(rad(transform.rotation_deg.x), Vector3f::UnitX());
  //         AngleAxisf rot_y(rad(transform.rotation_deg.y), Vector3f::UnitY());
  //         AngleAxisf rot_z(rad(transform.rotation_deg.z), Vector3f::UnitZ());
  //         Quaternionf q = rot_z * rot_y * rot_x;
  //         Affine3f rot_transform = Translation3f(-alignment_center) * q *
  //                                  Translation3f(alignment_center);

  //         // perform manual rotation
  //         pos_f = rot_transform * pos_f;
  //         ori_f = q * ori_f;

  //         // then alignment translation
  //         pos_f += alignment_center + aligned_position_offset;

  //         // perform our manual translation
  //         pos_f += Vector3f(transform.offset.x, transform.offset.y,
  //                           transform.offset.z);

  //         // specified axis flips
  //         if (transform.flip_x) {
  //           pos_f.x() = -pos_f.x();
  //           ori_f *= Quaternionf(AngleAxisf(M_PI, Vector3f::UnitX()));
  //         }
  //         if (transform.flip_y) {
  //           pos_f.y() = -pos_f.y();
  //           ori_f *= Quaternionf(AngleAxisf(M_PI, Vector3f::UnitY()));
  //         }
  //         if (transform.flip_z) {
  //           pos_f.z() = -pos_f.z();
  //           ori_f *= Quaternionf(AngleAxisf(M_PI, Vector3f::UnitZ()));
  //         }

  //         // and scaling
  //         pos_f *= transform.scale;

  //         position pos_out = {static_cast<short>(std::round(pos_f.x())),
  //                             static_cast<short>(std::round(pos_f.y())),
  //                             static_cast<short>(std::round(pos_f.z())), 0};

  //         skeleton[joint].first = {pos_out.x, pos_out.y, pos_out.z};
  //         skeleton[joint].second = {ori_f.w(), ori_f.x(), ori_f.y(), ori_f.z()};
  //       }
  //       _skeletons.push_back(skeleton);
  //     }
  //   }
  }
};

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(AzureKinectBodyTracking,
                        pc::devices::AzureKinectBodyTracking,
                        "net.pointcaster.k4abt/1.0")