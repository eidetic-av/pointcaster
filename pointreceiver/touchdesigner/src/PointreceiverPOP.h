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
#include <string>
using namespace TD;

// To get more help about these functions, look at POP_CPlusPlusBase.h
class PointreceiverPOP : public POP_CPlusPlusBase
{
public:

	PointreceiverPOP(const OP_NodeInfo* info, POP_Context* context);

	virtual ~PointreceiverPOP();

	virtual void	getGeneralInfo(POP_GeneralInfo*, const OP_Inputs*, void* reserved1) override;

	virtual void	execute(POP_Output*, const OP_Inputs*, void* reserved) override;

	virtual int32_t getNumInfoCHOPChans(void* reserved) override;

	virtual void	getInfoCHOPChan(int index, OP_InfoCHOPChan* chan, void* reserved) override;

	virtual bool	getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved) override;

	virtual void	getInfoDATEntries(int32_t index, int32_t nEntries,
									OP_InfoDATEntries* entries,
									void* reserved) override;

	virtual void	setupParameters(OP_ParameterManager* manager, void* reserved) override;

private:

	// example functions for generating a geometry, change them with any
	// fucntions and algorithm:

	void			cubeGeometry(POP_Output* output, float scale);
	void			mixedPrimitives(POP_Output* output);
	void			mixedPrimitivesComplex(POP_Output* output);
	void			gridGeometry(POP_Output* output);

	OP_SmartRef<POP_Buffer>		createTopologyInfoBuffer();
	OP_SmartRef<POP_Buffer>		createPointInfoBuffer();
	OP_SmartRef<POP_Buffer>		createGridInfoBuffer(uint32_t numDims);

	OP_SmartRef<POP_Buffer>		createBuffer(POP_BufferInfo createInfo);

	// We don't need to store this pointer, but we do for the example.
	// The OP_NodeInfo class store information about the node that's using
	// this instance of the class (like its name).
	const OP_NodeInfo* const	myNodeInfo;

	// In this example this value will be incremented each time the execute()
	// function is called, then passes back to the POP
	int32_t						myExecuteCount;

	// The context is used for things such as buffer memory management.
	// It is held inside the class instead of passed into the callbacks
	// since the class (or threads it spawns) can use the context to
	// do operations such as allocate memory outside of the callbacks.
	POP_Context* const			myContext;
};
