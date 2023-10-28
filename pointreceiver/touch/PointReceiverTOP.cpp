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

#include "PointReceiverTOP.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <limits>

#include <pointreceiver.h>

std::atomic<PointReceiverState> PointReceiverTOP::point_receiver_state = PointReceiverState::Inactive;
std::atomic<bool> PointReceiverTOP::quit_receiver_thread = false;

std::vector<TOPOutput_T> PointReceiverTOP::positions_buffer{};
std::vector<float> PointReceiverTOP::colors_buffer{};

int PointReceiverTOP::point_count = 0;
std::mutex PointReceiverTOP::buffers_access{};

void PointReceiverTOP::startReceiving(std::string point_caster_address) {
	point_receiver_state = PointReceiverState::Connecting;
	// run the connection process on a new thread
	// because we have some blocking processes
	std::thread initialise_networking{[&](std::string connect_address) {
		using namespace std::chrono_literals;
		startNetworkThread(connect_address);
		// wait until a valid frame comes through before considering
		// our state "connected"
		point_count = 0;
		while (point_count == 0 && !quit_receiver_thread) {
			point_count = getPointCount();
			std::this_thread::sleep_for(100ms);
		}
		point_receiver_state = PointReceiverState::Connected;
		while (!quit_receiver_thread) {

			buffers_access.lock();

			point_count = getPointCount();

			positions_buffer.resize(point_count);
			colors_buffer.resize(point_count);

			auto point_positions = getPointPositionsBuffer();
			auto point_colors = getPointColorsBuffer();

			auto local_positions_addr = positions_buffer.data();
			auto local_colors_addr = colors_buffer.data();

			std::memcpy(local_positions_addr, point_positions, point_count * sizeof(TOPOutput_T));
			std::memcpy(local_colors_addr, point_colors, point_count * sizeof(float));
			
			buffers_access.unlock();

			std::this_thread::sleep_for(3ms);
		}
		stopNetworkThread();
		point_receiver_state = PointReceiverState::Inactive;
		quit_receiver_thread = false;
	}, point_caster_address};
	initialise_networking.detach();
}

extern "C" {

DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::CPUMemWriteOnly;

	// The opType is the unique name for this TOP. It must start with a 
	// capital A-Z character, and all the following characters must lower case
	// or numbers (a-z, 0-9)
	info->customOPInfo.opType->setString("Bobpointreceiver");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("BoB Point Receiver");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("BPR");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("Matt Hughes");
	info->customOPInfo.authorEmail->setString("matth@eidetic.net.au");

	// This TOP works with 0 inputs connected
	info->customOPInfo.minInputs = 0;
	info->customOPInfo.maxInputs = 0;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context) {
	return new PointReceiverTOP(info);
}

DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context) {
	auto top_instance = (PointReceiverTOP*)instance;
	delete top_instance;
}

};

PointReceiverTOP::PointReceiverTOP(const OP_NodeInfo* info) : worker_thread(nullptr) {
	nodeInfo = info;
	quit_worker_thread = false;
	startWork = false;
	hostname = "192.168.1.10:9999";
}

PointReceiverTOP::~PointReceiverTOP() {
	if (worker_thread) {
		quit_worker_thread = true;
		// Incase the thread is sleeping waiting for a signal
		// to create more work, wake it up
		startMoreWork();
		if (worker_thread->joinable()) worker_thread->join();
		delete worker_thread;
	}
}

TOPOutputMode PointReceiverTOP::getOutputMode() {
	return node_output_mode;
}

void PointReceiverTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void* reserved1) {

	ginfo->cookEveryFrame = true;
	ginfo->memPixelType = OP_CPUMemPixelType::RGBA32Float;
}

bool PointReceiverTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs* inputs, void* reserved1) {
	format->redChannel = true;
	format->greenChannel = true;
	format->blueChannel = true;
	format->alphaChannel = false;
	format->bitsPerChannel = 32;
	format->numColorBuffers = 1;
	format->depthBits = 0;
	format->stencilBits = 0;
	return true;
}

void PointReceiverTOP::execute(TOP_OutputFormatSpecs* output,
						const OP_Inputs* inputs,
						TOP_Context *context,
						void* reserved1) {

	// update any input settings
	settingsLock.lock();
	auto active = inputs->getParInt("Active");
	hostname = inputs->getParString("Hostname");
	stacked_output_height = inputs->getParInt("Stackheight");
	auto output_mode_str = std::string(inputs->getParString("Outputmode"));
	settingsLock.unlock();

	if (output_mode_str == "Positions") node_output_mode = TOPOutputMode::Positions;
	else if (output_mode_str == "Colors") node_output_mode = TOPOutputMode::Colors;
	else node_output_mode = TOPOutputMode::Stacked;

	if (point_receiver_state == PointReceiverState::Inactive && active) 
		startReceiving(hostname);
	else if (point_receiver_state == PointReceiverState::Connected && !active)
		quit_receiver_thread = true;

	std::unique_lock<std::mutex> lock(buffers_access);
	int textureMemoryLocation = 0;
	float* mem = (float*)output->cpuPixelData[textureMemoryLocation];
	fillBuffer(mem, output->width, output->height, point_receiver_state, node_output_mode, stacked_output_height);
	output->newCPUPixelDataLocation = textureMemoryLocation;
}

void PointReceiverTOP::fillBuffer(void* mem, int width, int height, PointReceiverState point_receiver_state, TOPOutputMode output_mode, int stacked_output_height)
{
	constexpr auto empty_pixel = TOPOutput_T{ 0.f, 0.f, 0.f , 0.f};
	constexpr auto filled_pixel = TOPOutput_T{ 1.f, 1.f, 1.f , 1.f};

	auto pixel_count = width * height;
	auto texture_output = reinterpret_cast<TOPOutput_T*>(mem);
		
	// clear the entire texture first
	std::fill_n(texture_output, pixel_count, empty_pixel);

	//std::unique_lock<std::mutex> lock(buffers_access);

	if (output_mode == TOPOutputMode::Positions) {
		// load it up from our pointreceiver lib's positions buffer
		int i = 0;
		for (auto pos : positions_buffer) {
			auto pixel = &texture_output[i++];
			pixel->r = pos.r;
			pixel->g = pos.g;
			pixel->b = pos.b;
			pixel->a = 1.f;
			if (i >= pixel_count) break;
		}
	}
	else if (output_mode == TOPOutputMode::Colors) {
		// load it up from our pointreceiver lib's colors buffer
		int i = 0;
		for (auto color : colors_buffer) {
			auto pixel = &texture_output[i++];
			// input color is in 8-bits per channel,
			// output pixel is 32-bits per channel
			auto rgba = *reinterpret_cast<int32_t*>(&color);
			pixel->a = float(rgba >> 24) / 255.0;
			pixel->r = float((rgba & 0x00ff0000) >> 16) / 255.0;
			pixel->g = float((rgba & 0x0000ff00) >> 8) / 255.0;
			pixel->b = float(rgba & 0x000000ff) / 255.0;
			if (i>= pixel_count) break;
		}
	}
	else if (output_mode == TOPOutputMode::Stacked) {
		// stacked mode places position and color data in the same frame
		int i = 0;
		for (auto pos : positions_buffer) {
			if (i >= pixel_count) break;
			auto pixel = &texture_output[i++];
			pixel->r = pos.r;
			pixel->g = pos.g;
			pixel->b = pos.b;
			pixel->a = 1.f;
		}
		i = stacked_output_height * width;
		for (auto color : colors_buffer) {
			if (i>= pixel_count) break;
			auto pixel = &texture_output[i++];
			// input color is in 8-bits per channel,
			// output pixel is 32-bits per channel
			auto rgba = *reinterpret_cast<int32_t*>(&color);
			pixel->a = float(rgba >> 24) / 255.0;
			pixel->r = float((rgba & 0x00ff0000) >> 16) / 255.0;
			pixel->g = float((rgba & 0x0000ff00) >> 8) / 255.0;
			pixel->b = float(rgba & 0x000000ff) / 255.0;
		}
	}
}

void PointReceiverTOP::startMoreWork() {
	{
		std::unique_lock<std::mutex> lck(conditionLock);
		startWork = true;
	}
	condition.notify_one();
}

void PointReceiverTOP::waitForMoreWork() {
	std::unique_lock<std::mutex> lck(conditionLock);
	condition.wait(lck, [this]() { return this->startWork.load(); });
	startWork = false;
}


int32_t PointReceiverTOP::getNumInfoCHOPChans(void *reserved1) {
	return 0;
}

void PointReceiverTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void* reserved1) {
}

bool PointReceiverTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved1)
{
	infoSize->rows = 0;
	infoSize->cols = 0;
	infoSize->byColumn = false;
	return true;
}

void PointReceiverTOP::getInfoDATEntries(int32_t index,
								int32_t nEntries,
								OP_InfoDATEntries* entries,
								void *reserved1)
{

}

void PointReceiverTOP::setupParameters(OP_ParameterManager* manager, void *reserved1)
{
	//zmq address
	{
		OP_StringParameter sp;

		sp.name = "Hostname";
		sp.label = "Hostname";
		sp.defaultValue = "192.168.1.10:9999";

		OP_ParAppendResult res = manager->appendString(sp);
		assert(res == OP_ParAppendResult::Success);
	}
	
	// connection active toggle
	{
		OP_NumericParameter	np;

		np.name = "Active";
		np.label = "Active";
		
		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// output mode select
	{
		OP_StringParameter	sp;

		sp.name = "Outputmode";
		sp.label = "Output Mode";
		sp.defaultValue = "Positions";

		const char *names[] = {"Positions", "Colors", "Stacked"};
		const char *labels[] = {"Positions", "Colors", "Stacked"};

		OP_ParAppendResult res = manager->appendMenu(sp, 3, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

	// stacked output height
	{
		OP_NumericParameter np;

		np.name = "Stackheight";
		np.label = "Stacked output height";
		
		OP_ParAppendResult res = manager->appendInt(np);
		assert(res == OP_ParAppendResult::Success);
	}

}

