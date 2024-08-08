#include "orbbec_device.h"
#include "../../logger.h"
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Utils.hpp>
#include <thread>

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
	}

	// pass in device to create pipeline
	_ob_pipeline = std::make_shared<ob::Pipeline>(_ob_device);

	// Create Config for configuring Pipeline work
	_ob_config = std::make_shared<ob::Config>();

	_ob_config->setAlignMode(ALIGN_D2C_HW_MODE);
	_ob_config->setFrameAggregateOutputMode(
		OB_FRAME_AGGREGATE_OUTPUT_FULL_FRAME_REQUIRE);

	constexpr auto fps = 30;
	constexpr auto color_width = 1280;
	constexpr auto color_height = 720;

	auto color_profiles = _ob_pipeline->getStreamProfileList(OB_SENSOR_COLOR);
	auto color_profile = color_profiles->getVideoStreamProfile(
		color_width, color_height, OB_FORMAT_ANY, fps);

	_ob_config->enableStream(color_profile);

	// DEPTH_MODE_NFOV_UNBINNED
	auto depth_width = 640;
	auto depth_height = 576;
	if (_config.depth_mode == 1)
	{
		// DEPTH_MODE_WFOV_2X2BINNED
		depth_width = 512;
		depth_height = 512;
	}

	auto depth_profiles =
		_ob_pipeline->getD2CDepthProfileList(color_profile, ALIGN_D2C_HW_MODE);
	auto depth_profile = depth_profiles->getVideoStreamProfile(
		depth_width, depth_height, OB_FORMAT_ANY, fps);

	_ob_config->enableStream(depth_profile);

	init_device_memory(color_profile->width() * color_profile->height());

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

	_ob_point_cloud = std::make_unique<ob::PointCloudFilter>();
	_ob_point_cloud->setCameraParam(_ob_pipeline->getCameraParam());
	_ob_point_cloud->setCreatePointFormat(OB_FORMAT_RGB_POINT);

	_running_pipeline = true;
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
		if (_config.depth_mode != last_config.depth_mode) {
			restart();
		}
	};
}

} // namespace pc::devices
