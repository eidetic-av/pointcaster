#include "orbbec_device.h"
#include "../../logger.h"
#include <fmt/format.h>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Utils.hpp>
#include <oneapi/tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace pc::devices {

void OrbbecDevice::init_context()
{
	_ob_ctx = std::make_unique<ob::Context>();
	_ob_ctx->enableNetDeviceEnumeration(true);
}

void OrbbecDevice::discover_devices()
{
	discovering_devices = true;
	discovered_devices.clear();
	_ob_device_list = _ob_ctx->queryDeviceList();
	auto count = _ob_device_list->deviceCount();
	for (int i = 0; i < count; i++)
	{
		auto device = _ob_device_list->getDevice(i);
		auto info = device->getDeviceInfo();
		if (strcmp(info->connectionType(),"Ethernet") == 0)
		{
			discovered_devices.push_back({ info->ipAddress(), info->serialNumber() });
		}
	}
	discovering_devices = false;
}

OrbbecDevice::OrbbecDevice(DeviceConfiguration &config)
    : _config(config) {

  parameters::declare_parameters(_config.id, _config);

  _ip = std::string(_config.id.begin() + 3, _config.id.end());
  std::replace(_ip.begin(), _ip.end(), '_', '.');

  OrbbecDevice::attached_devices.push_back(std::ref(*this));
  start();
}

OrbbecDevice::~OrbbecDevice() {
	stop_pipeline();

	parameters::unbind_parameters(_config.id);

	// remove this instance from attached devices
	std::erase_if(attached_devices, [this](auto& device) {
		return &(device.get()) == this;
	});
	pc::logger->debug("Destroyed OrbbecDevice");
}

void OrbbecDevice::start() {
	_initialisation_thread = std::jthread([this](std::stop_token stop_token) {
		// TODO stop_token is currently unused,
		// should pass it in and check at different initialisation thread steps
		start_pipeline();
	});
}

void OrbbecDevice::stop() {
	_initialisation_thread = std::jthread([this](std::stop_token stop_token) {
		// TODO stop_token is currently unused,
		// should pass it in and check at different initialisation thread steps
		stop_pipeline();
	});
}

void OrbbecDevice::restart() {
	_initialisation_thread = std::jthread([this](std::stop_token stop_token) {
		// TODO stop_token is currently unused,
		// should pass it in and check at different initialisation thread steps
		stop_pipeline();
		start_pipeline();
	});
}

void OrbbecDevice::start_pipeline() {
	pc::logger->info("Initialising OrbbecDevice at {}", _ip);

	constexpr uint16_t net_device_port = 8090; // femto mega default

	bool existing_device = false;

	_ob_device_list = _ob_ctx->queryDeviceList();
	auto count = _ob_device_list->deviceCount();
	for (int i = 0; i < count; i++)
	{
		auto device = _ob_device_list->getDevice(i);
		auto info = device->getDeviceInfo();
		if (strcmp(info->ipAddress(), _ip.c_str()) == 0)
		{
			_ob_device = device;
			existing_device = true;
		}
	}

	if (!existing_device) {
		try {
			_ob_device = _ob_ctx->createNetDevice(_ip.data(), net_device_port);
		}
		catch (const ob::Error& e) {
			pc::logger->error("Failed to create OrbbecDevice at {}:{}", _ip,
				net_device_port);
			return;
		}
		catch (...) {
			pc::logger->error("Unknown error creating Orrbec NetDevice");
		}
	}

	_ob_pipeline = std::make_shared<ob::Pipeline>(_ob_device);
	_ob_config = std::make_shared<ob::Config>();
  _ob_point_cloud = std::make_unique<ob::PointCloudFilter>();

  _ob_xy_tables.reset();
  _xy_table_data.reset();
  _depth_positions_buffer.reset();
  _depth_frame_buffer.reset();

  constexpr auto fps = 30;
  constexpr auto color_width = 1280;
  constexpr auto color_height = 720;

  // Default to using Narrow FOV
  auto depth_width = 640;
  auto depth_height = 576;
  // Wide FOV depth mode
  if (_config.depth_mode == 1)
  {
    depth_width = 512;
    depth_height = 512;
  }


  // default acquisition mode is RGBD
  if (_config.acquisition_mode == 0) {

    _ob_config->setAlignMode(ALIGN_D2C_HW_MODE);
    _ob_config->setFrameAggregateOutputMode(
      OB_FRAME_AGGREGATE_OUTPUT_FULL_FRAME_REQUIRE);

    auto color_profiles = _ob_pipeline->getStreamProfileList(OB_SENSOR_COLOR);
    auto color_profile = color_profiles->getVideoStreamProfile(
      color_width, color_height, OB_FORMAT_ANY, fps);

    _ob_config->enableStream(color_profile);

    auto depth_profiles =
      _ob_pipeline->getD2CDepthProfileList(color_profile, ALIGN_D2C_HW_MODE);
    auto depth_profile = depth_profiles->getVideoStreamProfile(
      depth_width, depth_height, OB_FORMAT_ANY, fps);

    _ob_config->enableStream(depth_profile);

    if (!init_device_memory(color_profile->width() * color_profile->height())) {
      pc::logger->error("Failed to initialise GPU memory for Orbbec pipeline");
      return;
    };

    _ob_pipeline->start(_ob_config, [&](std::shared_ptr<ob::FrameSet> frame_set) {
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

    _ob_point_cloud->setCameraParam(_ob_pipeline->getCameraParam());
    _ob_point_cloud->setCreatePointFormat(OB_FORMAT_RGB_POINT);

    _running_pipeline = true;
  }

  // depth_only acquisition_mode
  else if (_config.acquisition_mode == 1) {

    auto depth_profiles = _ob_pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
    auto depth_profile = depth_profiles->getVideoStreamProfile(depth_width, depth_height, OB_FORMAT_Y16, fps);

    _ob_config->enableStream(depth_profile);
    _ob_config->setAlignMode(ALIGN_DISABLE);

    if (!init_device_memory(depth_width * depth_height)) {
      pc::logger->error("Failed to initialise GPU memory for Orbbec pipeline");
      return;
    }

    _point_buffer.resize(depth_width * depth_height);

    auto camera_param = _ob_pipeline->getCameraParam();
    _ob_point_cloud->setCameraParam(camera_param);

    auto calibration_param = _ob_pipeline->getCalibrationParam(_ob_config);
    auto stream_profile = depth_profile->as<ob::VideoStreamProfile>();

    uint32_t xy_table_size = depth_width * depth_height * 2;
    _xy_table_data = std::vector<float>(xy_table_size);
    _ob_xy_tables = OBXYTables{};
    bool table_init = ob::CoordinateTransformHelper::transformationInitXYTables(
      calibration_param, OB_SENSOR_DEPTH, 
      (*_xy_table_data).data(), &xy_table_size, &(*_ob_xy_tables));

    if (!table_init) {
      pc::logger->error("Failed to initialise orbbec transformation tables");
      return;
    }

    const auto frame_size = depth_height * depth_width;
    _depth_positions_buffer = std::vector<OBPoint3f>(frame_size);
    _depth_frame_buffer = std::vector<OBColorPoint>(frame_size);

    _ob_pipeline->start(_ob_config, [this, frame_size]
                        (std::shared_ptr<ob::FrameSet> frame_set) {
      if (frame_set == nullptr || frame_set->depthFrame() == nullptr) return;

      auto depth_frame = frame_set->depthFrame();
      auto* depth_data = static_cast<uint16_t*>(depth_frame->data());

      auto& depth_positions_buffer = (*_depth_positions_buffer);
      auto& depth_frame_buffer = (*_depth_frame_buffer);

      ob::CoordinateTransformHelper::transformationDepthToPointCloud(&(*_ob_xy_tables), depth_data, depth_positions_buffer.data());
      
      tbb::parallel_for(
        tbb::blocked_range<size_t>(0, frame_size),
        [&](auto& r) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            auto pos = depth_positions_buffer[i];
            depth_frame_buffer[i] = OBColorPoint { pos.x, pos.y, pos.z, 255, 255, 255 };
        }
      });

      {
        std::lock_guard lock(_point_buffer_access);
        std::swap(_point_buffer, depth_frame_buffer);
        _buffer_updated = true;
      }

    });

    _running_pipeline = true;
  }
}

void OrbbecDevice::stop_pipeline() {
	pc::logger->info("Closing OrbbecDevice {}", _ip);
	_running_pipeline = false;
	_ob_pipeline->stop();
	free_device_memory();
}

void OrbbecDevice::draw_imgui_controls() {
	auto running = _running_pipeline;
	ImGui::Dummy({ 10, 0 }); ImGui::SameLine();
	if (running) ImGui::BeginDisabled();
	if (ImGui::Button("Start")) start();
	if (running) ImGui::EndDisabled();
	if (!running) ImGui::BeginDisabled();
	ImGui::SameLine();
	ImGui::Dummy({ 10, 10 });
	ImGui::SameLine();
	if (ImGui::Button("Stop")) stop();
	if (!running) ImGui::EndDisabled();
	DeviceConfiguration last_config = _config;
	if (pc::gui::draw_parameters(_config.id)) {
    if (_config.depth_mode != last_config.depth_mode
      || _config.acquisition_mode != last_config.acquisition_mode) {
      restart();
		}
	};
}

} // namespace pc::devices
