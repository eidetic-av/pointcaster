#include "orbbec_device.h"
#include "../../logger.h"
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Utils.hpp>

namespace pc::devices {

OrbbecDevice::OrbbecDevice(DeviceConfiguration &config, std::string_view id)
    : _config(config), _id(id.begin(), id.end()) {

  parameters::declare_parameters(_id, _config);

  _ip = std::string(_id.begin() + 3, _id.end());
  std::replace(_ip.begin(), _ip.end(), '_', '.');

  pc::logger->info("Initialising OrbbecDevice at {}", _ip);

  _ob_ctx = std::make_unique<ob::Context>();
  constexpr uint16_t net_device_port = 8090; // femto mega default

  std::shared_ptr<ob::Device> device;
  try {
    device = _ob_ctx->createNetDevice(_ip.data(), net_device_port);
  } catch (const ob::Error &e) {
    pc::logger->error("Failed to create OrbbecDevice at {}:{}", _ip,
		      net_device_port);
    return;
  }

  // pass in device to create pipeline
  _ob_pipeline = std::make_shared<ob::Pipeline>(device);

  // Create Config for configuring Pipeline work
  std::shared_ptr<ob::Config> ob_config = std::make_shared<ob::Config>();

  ob_config->setAlignMode(ALIGN_D2C_HW_MODE);
  ob_config->setFrameAggregateOutputMode(
      OB_FRAME_AGGREGATE_OUTPUT_FULL_FRAME_REQUIRE);

  const auto fps = 30;

  const auto color_width = 1280;
  const auto color_height = 720;

  auto color_profiles = _ob_pipeline->getStreamProfileList(OB_SENSOR_COLOR);
  auto color_profile = color_profiles->getVideoStreamProfile(
      color_width, color_height, OB_FORMAT_ANY, fps);

  ob_config->enableStream(color_profile);

  // DEPTH_MODE_WFOV_2X2BINNED
  // constexpr auto depth_width = 512;
  // constexpr auto depth_height = 512;
  // DEPTH_MODE_NFOV_UNBINNED
  const auto depth_width = 640;
  const auto depth_height = 576;

  auto depth_profiles =
      _ob_pipeline->getD2CDepthProfileList(color_profile, ALIGN_D2C_HW_MODE);
  auto depth_profile = depth_profiles->getVideoStreamProfile(
      depth_width, depth_height, OB_FORMAT_ANY, fps);

  ob_config->enableStream(depth_profile);

  init_device_memory(color_profile->width() * color_profile->height());

  _ob_pipeline->start(ob_config, [&](std::shared_ptr<ob::FrameSet> frame_set) {
    auto color_frame = frame_set->colorFrame();
    auto point_count = color_frame->width() * color_frame->height();
    auto point_cloud = _ob_point_cloud->process(frame_set);
    if (point_cloud && point_cloud->data()) {
      std::lock_guard lock(_point_buffer_access);
      _point_buffer.resize(point_count);
      std::memcpy(_point_buffer.data(), point_cloud->data(),
		  point_count * sizeof(OBColorPoint));
    }
    _buffer_updated = true;
  });

  _ob_point_cloud = std::make_unique<ob::PointCloudFilter>();
  _ob_point_cloud->setCameraParam(_ob_pipeline->getCameraParam());
  _ob_point_cloud->setCreatePointFormat(OB_FORMAT_RGB_POINT);

  OrbbecDevice::attached_devices.push_back(std::ref(*this));
}

OrbbecDevice::~OrbbecDevice() {
  pc::logger->info("Closing OrbbecDevice {}", _ip);
  _ob_pipeline->stop();
  free_device_memory();
}

void OrbbecDevice::draw_imgui_controls() {
  pc::gui::draw_parameters(_id);
}

} // namespace pc::devices
