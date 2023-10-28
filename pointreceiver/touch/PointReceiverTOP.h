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

#include "TOP_CPlusPlusBase.h"
#include "FrameQueue.h"
#include <thread>
#include <atomic>
#include <string>
#include "pointreceiver.h"

enum class PointReceiverState {
	Inactive, Connecting, Connected
};

enum class TOPOutputMode {
	Positions, Colors, Stacked
};

struct TOPOutput_T {
	float r, g, b, a = 0.f;
};

class PointReceiverTOP : public TOP_CPlusPlusBase
{
public:

	static std::atomic<PointReceiverState> point_receiver_state;
	static std::atomic<bool> quit_receiver_thread;
	static std::vector<TOPOutput_T> positions_buffer;
	static std::vector<float> colors_buffer;
	static int point_count;
	static std::mutex buffers_access;

	static void startReceiving(std::string point_caster_address);

	PointReceiverTOP(const OP_NodeInfo *info);
	virtual ~PointReceiverTOP();

	TOPOutputMode getOutputMode();

	virtual void		getGeneralInfo(TOP_GeneralInfo *, const OP_Inputs*, void*) override;
	virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void*) override;


	virtual void		execute(TOP_OutputFormatSpecs*,
							const OP_Inputs*,
							TOP_Context* context,
							void* reserved1) override;

	static void			fillBuffer(void* mem, int width, int height, PointReceiverState point_receiver_state, TOPOutputMode output_mode, int stacked_output_height);


	virtual int32_t		getNumInfoCHOPChans(void *reserved1) override;
	virtual void		getInfoCHOPChan(int32_t index,
								OP_InfoCHOPChan *chan, void* reserved1) override;

	virtual bool		getInfoDATSize(OP_InfoDATSize *infoSize, void *reserved1) override;
	virtual void		getInfoDATEntries(int32_t index,
									int32_t nEntries,
									OP_InfoDATEntries *entries,
									void *reserved1) override;

	virtual void		setupParameters(OP_ParameterManager *manager, void *reserved1) override;

	void				waitForMoreWork();

private:

	void				startMoreWork();

	const OP_NodeInfo*	nodeInfo;

	TOPOutputMode node_output_mode;

	std::mutex			settingsLock;
	const char*			hostname;
	
	int stacked_output_height = 256;

	// Used for threading example
	// Search for #define THREADING_EXAMPLE to enable that example
	FrameQueue			frameQueue;
	// thread used to fill our output buffer
	std::thread*		worker_thread;
	std::atomic<bool>	quit_worker_thread;

	std::condition_variable	condition;
	std::mutex			conditionLock;
	std::atomic<bool>	startWork;

};
