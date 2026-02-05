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

#include "PointreceiverPOP.h"

#include <stdio.h>
#include <array>
#include <vector>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <pointreceiver.h>

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

	DLLEXPORT
	void
	FillPOPPluginInfo(POP_PluginInfo *info)
	{
		// Always set this to POPCPlusPlusAPIVersion
		if (!info->setAPIVersion(POPCPlusPlusAPIVersion))
			return;

		// The opType is the unique name for this TOP. It must start with a
		// capital A-Z character, and all the following characters must lower case
		// or numbers (a-z, 0-9)
		info->customOPInfo.opType->setString("Pointreceiver");

		// The opLabel is the text that will show up in the OP Create Dialog
		info->customOPInfo.opLabel->setString("Simple Shapes");

		// Will be turned into a 3 letter icon on the nodes
		info->customOPInfo.opIcon->setString("SSP");

		// Information about the author of this OP
		info->customOPInfo.authorName->setString("Author Name");
		info->customOPInfo.authorEmail->setString("email@email.com");

		// This POP works with 0 or 1 inputs
		info->customOPInfo.minInputs = 0;
		info->customOPInfo.maxInputs = 1;

		// Custom website URL that the Operator Help can point to
		info->customOPInfo.opHelpURL->setString("yourwebsiteurl.com");

	}

	DLLEXPORT
	POP_CPlusPlusBase*
	CreatePOPInstance(const OP_NodeInfo* info, POP_Context* context)
	{
		// Return a new instance of your class every time this is called.
		// It will be called once per POP that is using the .dll
		return new PointreceiverPOP(info, context);
	}

	DLLEXPORT
	void
	DestroyPOPInstance(POP_CPlusPlusBase* instance)
	{
		// Delete the instance here, this will be called when
		// Touch is shutting down, when the POP using that instance is deleted, or
		// if the POP loads a different DLL
		delete (PointreceiverPOP*)instance;
	}

};

PointreceiverPOP::PointreceiverPOP(const OP_NodeInfo* info, POP_Context* context) :
	myNodeInfo(info),
	myContext(context)
{
	myExecuteCount = 0;
}

PointreceiverPOP::~PointreceiverPOP()
{

}

void
PointreceiverPOP::getGeneralInfo(POP_GeneralInfo* ginfo, const OP_Inputs* inputs, void* reserved)
{
	// This will cause the node to cook every frame
	ginfo->cookEveryFrameIfAsked = false;
}

//-----------------------------------------------------------------------------------------------------
//										Generate a geometry on CPU
//-----------------------------------------------------------------------------------------------------

template <class T, size_t N>
static int64_t
getArrayByteSize(const std::array<T, N>& arr)
{
	return N * sizeof(T);
}

OP_SmartRef<POP_Buffer>
PointreceiverPOP::createBuffer(POP_BufferInfo createInfo)
{
	createInfo.location = POP_BufferLocation::CPU;
	createInfo.mode = POP_BufferMode::ReadWrite;
	OP_SmartRef<POP_Buffer> buf = myContext->createBuffer(createInfo, nullptr);
	void* data = (POP_PointInfo*)buf->getData(nullptr);
	memset(data, 0, createInfo.size);
	return buf;
}

OP_SmartRef<POP_Buffer>
PointreceiverPOP::createTopologyInfoBuffer()
{
	POP_BufferInfo createInfo;
	createInfo.usage = POP_BufferUsage::TopologyInfoBuffer;
	createInfo.size = sizeof(POP_TopologyInfo);

	return createBuffer(createInfo);
}

OP_SmartRef<POP_Buffer>
PointreceiverPOP::createPointInfoBuffer()
{
	POP_BufferInfo createInfo;
	createInfo.usage = POP_BufferUsage::PointInfoBuffer;
	createInfo.size = sizeof(POP_PointInfo);

	return createBuffer(createInfo);
}

OP_SmartRef<POP_Buffer>
PointreceiverPOP::createGridInfoBuffer(uint32_t numDims)
{
	POP_BufferInfo createInfo;
	createInfo.usage = POP_BufferUsage::GridInfoBuffer;
	createInfo.size = POP_GridInfo::getRequiredSize(numDims);

	return createBuffer(createInfo);
}

// This examples is a single set of triangles, using a mix of point, vertex and primitive attributes.
void
PointreceiverPOP::cubeGeometry(POP_Output* output, float scale)
{
	POP_SetBufferInfo sinfo;

	POP_BufferInfo attBufCreateInfo;
	uint64_t posSize = 8 * sizeof(Position);
	attBufCreateInfo.size = posSize;
	OP_SmartRef<POP_Buffer>	posBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!posBuf)
		return;
	Position* posData = (Position*)posBuf->getData(nullptr);

	// front
	posData[0] = Position(1.0f*scale, -1.0f, 1.0f),
	posData[1] = Position(3.0f*scale, -1.0f, 1.0f),
	posData[2] = Position(3.0f*scale, 1.0f, 1.0f),
	posData[3] = Position(1.0f*scale, 1.0f, 1.0f),
	// back
	posData[4] = Position(1.0f*scale, -1.0f, -1.0f);
	posData[5] = Position(3.0f*scale, -1.0f, -1.0f);
	posData[6] = Position(3.0f*scale, 1.0f, -1.0f);
	posData[7] = Position(1.0f*scale, 1.0f, -1.0f);

	POP_AttributeInfo posInfo;
	posInfo.numComponents = 3;
	posInfo.type = POP_AttributeType::Float;
	posInfo.name = "P";
	posInfo.attribClass = POP_AttributeClass::Point;
	output->setAttribute(&posBuf, posInfo, sinfo, nullptr);

	// Use vertex attribute for the normal, since we need different normals for
	// vertices, to face the direction of the side. They can't be point normals
	// for a cube.
	std::array<Vector, 36> normal =
	{
		// front
		Vector(0.0f, 0.0f, 1.0f),
		Vector(0.0f, 0.0f, 1.0f),
		Vector(0.0f, 0.0f, 1.0f),
		Vector(0.0f, 0.0f, 1.0f),
		Vector(0.0f, 0.0f, 1.0f),
		Vector(0.0f, 0.0f, 1.0f),

		// right
		Vector(1.0f, 0.0f, 0.0f),
		Vector(1.0f, 0.0f, 0.0f),
		Vector(1.0f, 0.0f, 0.0f),
		Vector(1.0f, 0.0f, 0.0f),
		Vector(1.0f, 0.0f, 0.0f),
		Vector(1.0f, 0.0f, 0.0f),

		// back
		Vector(0.0f, 0.0f, -1.0f),
		Vector(0.0f, 0.0f, -1.0f),
		Vector(0.0f, 0.0f, -1.0f),
		Vector(0.0f, 0.0f, -1.0f),
		Vector(0.0f, 0.0f, -1.0f),
		Vector(0.0f, 0.0f, -1.0f),

		// left
		Vector(-1.0f, 0.0f, 0.0f),
		Vector(-1.0f, 0.0f, 0.0f),
		Vector(-1.0f, 0.0f, 0.0f),
		Vector(-1.0f, 0.0f, 0.0f),
		Vector(-1.0f, 0.0f, 0.0f),
		Vector(-1.0f, 0.0f, 0.0f),

		// bottom
		Vector(0.0f, -1.0f, 0.0f),
		Vector(0.0f, -1.0f, 0.0f),
		Vector(0.0f, -1.0f, 0.0f),
		Vector(0.0f, -1.0f, 0.0f),
		Vector(0.0f, -1.0f, 0.0f),
		Vector(0.0f, -1.0f, 0.0f),

		// top
		Vector(0.0f, 1.0f, 0.0f),
		Vector(0.0f, 1.0f, 0.0f),
		Vector(0.0f, 1.0f, 0.0f),
		Vector(0.0f, 1.0f, 0.0f),
		Vector(0.0f, 1.0f, 0.0f),
		Vector(0.0f, 1.0f, 0.0f),
	};

	int64_t normalSize = getArrayByteSize(normal);
	attBufCreateInfo.size = normalSize;
	OP_SmartRef<POP_Buffer>	normalBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!normalBuf)
		return;
	memcpy(normalBuf->getData(nullptr), normal.data(), normalSize);

	POP_AttributeInfo normalInfo;
	normalInfo.numComponents = 3;
	normalInfo.type = POP_AttributeType::Float;
	normalInfo.qualifier = POP_AttributeQualifier::Direction;
	normalInfo.name = "N";
	normalInfo.attribClass = POP_AttributeClass::Vertex;
	output->setAttribute(&normalBuf, normalInfo, sinfo, nullptr);

	// Use primitive attribute for the color.
	// One attribute per triangle
	std::array<Color, 12> color =
	{
		// front
		Color(1.0f, 0.0f, 0.0f, 1.0f),
		Color(1.0f, 0.0f, 0.0f, 1.0f),

		// right
		Color(0.0f, 1.0f, 0.0f, 1.0f),
		Color(0.0f, 1.0f, 0.0f, 1.0f),

		// back
		Color(0.0f, 0.0f, 1.0f, 1.0f),
		Color(0.0f, 0.0f, 1.0f, 1.0f),

		// left
		Color(1.0f, 1.0f, 1.0f, 1.0f),
		Color(1.0f, 1.0f, 1.0f, 1.0f),

		// bottom
		Color(1.0f, 1.0f, 0.0f, 1.0f),
		Color(1.0f, 1.0f, 0.0f, 1.0f),

		// top
		Color(1.0f, 0.0f, 1.0f, 1.0f),
		Color(1.0f, 0.0f, 1.0f, 1.0f),
	};

	int64_t colorSize = getArrayByteSize(color);
	attBufCreateInfo.size = colorSize;
	OP_SmartRef<POP_Buffer>	colorBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!colorBuf)
		return;
	memcpy(colorBuf->getData(nullptr), color.data(), colorSize);

	POP_AttributeInfo colorInfo;
	colorInfo.numComponents = 4;
	colorInfo.type = POP_AttributeType::Float;
	colorInfo.name = "Color";
	colorInfo.attribClass = POP_AttributeClass::Primitive;
	// This is needed so the attribute is treated properly as a color for
	// color space correctness.
	colorInfo.qualifier = POP_AttributeQualifier::Color;
	output->setAttribute(&colorBuf, colorInfo, sinfo, nullptr);

	// indices for the triangle primitives

	std::array<int32_t, 36> indices =
	{
		// front
		0, 1, 2,
		2, 3, 0,
		// right
		1, 5, 6,
		6, 2, 1,
		// back
		7, 6, 5,
		5, 4, 7,
		// left
		4, 0, 3,
		3, 7, 4,
		// bottom
		4, 5, 1,
		1, 0, 4,
		// top
		3, 2, 6,
		6, 7, 3,
	};

	int64_t indexSize = getArrayByteSize(indices);
	POP_BufferInfo indexBufCreateInfo;
	indexBufCreateInfo.usage = POP_BufferUsage::IndexBuffer;
	indexBufCreateInfo.size = indexSize;
	OP_SmartRef<POP_Buffer>	indexBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!indexBuf)
		return;
	memcpy(indexBuf->getData(nullptr), indices.data(), indexSize);

	POP_IndexBufferInfo indexInfo;
	indexInfo.type = POP_IndexType::UInt32;
	output->setIndexBuffer(&indexBuf, indexInfo, sinfo, nullptr);

	POP_InfoBuffers infoBufs;

	infoBufs.pointInfo = createPointInfoBuffer();
	POP_PointInfo* pointInfo = (POP_PointInfo*)infoBufs.pointInfo->getData(nullptr);
	pointInfo->numPoints = 8;

	infoBufs.topoInfo = createTopologyInfoBuffer();
	POP_TopologyInfo* topoInfo = (POP_TopologyInfo*)infoBufs.topoInfo->getData(nullptr);
	topoInfo->trianglesCount = (uint32_t)indices.size() / 3;

	output->setInfoBuffers(&infoBufs, sinfo, nullptr);
}

// This mode shows how multiple different types of primitives can be mixed inside the same output.
// Only position is provided
void
PointreceiverPOP::mixedPrimitives(POP_Output* output)
{
	POP_SetBufferInfo sinfo;

	// to generate a geometry:
	// addPoint() is the first function to be called.
	// then we can add normals, colors, and any custom attributes for the points
	// last function to be called is addLines()

	// Generate some line strips
	// Points 0-8
	std::array<Position, 9> lineStrip0Pos =
	{
		Position(-0.8f, 0.0f, 1.0f),
		Position(-0.6f, 0.4f, 1.0f),
		Position(-0.4f, 0.8f, 1.0f),
		Position(-0.2f, 0.4f, 1.0f),
		Position(0.0f,  0.0f, 1.0f),
		Position(0.2f, -0.4f, 1.0f),
		Position(0.4f, -0.8f, 1.0f),
		Position(0.6f, -0.4f, 1.0f),
		Position(0.8f,  0.0f, 1.0f),
	};

	// Points 9-16
	std::array<Position, 8> lineStrip1Pos =
	{
		Position(-0.8f, 0.2f, 1.0f),
		Position(-0.6f, 0.6f, 1.0f),
		Position(-0.4f, 1.0f, 1.0f),
		Position(-0.2f, 0.6f, 1.0f),
		Position(0.0f,  0.2f, 1.0f),
		Position(0.2f, -0.2f, 1.0f),
		Position(0.4f, -0.6f, 1.0f),
		Position(0.6f, -0.2f, 1.0f),
	};

	// This is a geometry made up of a quad and a triangle, that share some points
	// Points 17-21
	std::array<Position, 5> triAndQuadPos =
	{
		Position(1.0f, 0.0f, 0.0f),
		Position(1.0f, 1.0f, 0.0f),
		Position(0.0f, 1.0f, 0.0f),
		Position(0.0f, 0.0f, 0.0f),
		Position(1.5f, 0.5f, 0.0f),
	};

	// Points 22-26
	std::array<Position, 5> pointPrimsPos =
	{
		Position(1.0f, 0.0f, 2.0f),
		Position(1.0f, 1.0f, 2.0f),
		Position(0.0f, 1.0f, 2.0f),
		Position(0.0f, 0.0f, 2.0f),
		Position(1.5f, 0.5f, 2.0f),
	};

	POP_BufferInfo attBufCreateInfo;
	uint64_t posSize = getArrayByteSize(lineStrip0Pos) + getArrayByteSize(lineStrip1Pos) + getArrayByteSize(triAndQuadPos) + getArrayByteSize(pointPrimsPos);
	attBufCreateInfo.size = posSize;
	OP_SmartRef<POP_Buffer>	posBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!posBuf)
		return;
	Position* posData = (Position*)posBuf->getData(nullptr);

	auto copyArray =
		[&posData](const auto& arr)
		{
			memcpy(posData, arr.data(), getArrayByteSize(arr));
			posData += arr.size();
		};

	copyArray(lineStrip0Pos);
	copyArray(lineStrip1Pos);
	copyArray(triAndQuadPos);
	copyArray(pointPrimsPos);

	POP_AttributeInfo posInfo;
	posInfo.numComponents = 3;
	posInfo.type = POP_AttributeType::Float;
	posInfo.name = "P";
	posInfo.attribClass = POP_AttributeClass::Point;
	output->setAttribute(&posBuf, posInfo, sinfo, nullptr);

	////////////////////////////////////////////

	// Notice that altough the different primitive types need to appear in the index buffer in the same
	// order that they appear in the POP_PrimitiveInfo class:
	// triangles, quads, lineStrips, lines, pointPrimitives
	// they can reference point indices anywhere in the buffer.
	std::array<uint32_t, 31> indices =
	{
		// triangle
		18, 17, 21,

		// quad
		17, 18, 19, 20,

		// lineStrip0
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		// line strip restart index
		0xFFFFFFFFU,

		// lineStrip1
		9, 10, 11, 12, 13, 14, 15, 16,
		// line strip restart index
		0xFFFFFFFFU,

		// point primitives
		22, 23, 24, 25, 26
	};

	int64_t indexSize = getArrayByteSize(indices);
	POP_BufferInfo indexBufCreateInfo;
	indexBufCreateInfo.usage = POP_BufferUsage::IndexBuffer;
	indexBufCreateInfo.size = indexSize;
	OP_SmartRef<POP_Buffer>	indexBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!indexBuf)
		return;
	memcpy(indexBuf->getData(nullptr), indices.data(), indexSize);

	POP_IndexBufferInfo indexInfo;
	indexInfo.type = POP_IndexType::UInt32;
	output->setIndexBuffer(&indexBuf, indexInfo, sinfo, nullptr);

	////////////////////////////////////////////

	// Each line strip requires two entries in this buffer.
	// One is the start index, the second is the number of points in the strip, including the restart index
	uint64_t lineStripsInfoSize = 2 * 2 * sizeof(uint32_t);
	POP_BufferInfo lineStripsInfoCreateInfo;
	lineStripsInfoCreateInfo.usage = POP_BufferUsage::LineStripsInfoBuffer;
	lineStripsInfoCreateInfo.size = lineStripsInfoSize;

	OP_SmartRef<POP_Buffer>	lineStripsInfoBuf = myContext->createBuffer(lineStripsInfoCreateInfo, nullptr);
	if (!lineStripsInfoBuf)
		return;
	uint32_t* lineStripsInfoData = (uint32_t*)lineStripsInfoBuf->getData(nullptr);
	// First index location in the index buffer of the line strip, starting where the line strips being
	lineStripsInfoData[0] = 0;
	// number of indices in the line strip, including the restart index
	lineStripsInfoData[1] = (uint32_t)(lineStrip0Pos.size() + 1);
	// Don't read from the lineStripsInfoData[1] memory, since that is not allocated as readable.
	lineStripsInfoData[2] = (uint32_t)(lineStrip0Pos.size() + 1);
	lineStripsInfoData[3] = (uint32_t)(lineStrip1Pos.size() + 1);

	POP_InfoBuffers infoBufs;

	infoBufs.pointInfo = createPointInfoBuffer();
	POP_PointInfo* pointInfo = (POP_PointInfo*)infoBufs.pointInfo->getData(nullptr);
	pointInfo->numPoints = uint32_t(posSize / sizeof(Position));

	infoBufs.topoInfo = createTopologyInfoBuffer();
	POP_TopologyInfo* topoInfo = (POP_TopologyInfo*)infoBufs.topoInfo->getData(nullptr);

	topoInfo->trianglesStartIndex = 0;
	topoInfo->trianglesCount = 1;

	topoInfo->quadsStartIndex = topoInfo->trianglesStartIndex + topoInfo->trianglesCount * 3;
	topoInfo->quadsCount = 1;

	topoInfo->lineStripsStartIndex = topoInfo->quadsStartIndex + topoInfo->quadsCount * 4;
	// +2 for the restart indices
	topoInfo->lineStripsNumVertices = (uint32_t)(lineStrip0Pos.size() + lineStrip1Pos.size() + 2);
	topoInfo->lineStripsCount = 2;

	topoInfo->pointPrimitivesStartIndex = topoInfo->lineStripsStartIndex + topoInfo->lineStripsNumVertices;
	topoInfo->pointPrimitivesCount = (uint32_t)pointPrimsPos.size();

	////////////////////////////////////////////

	uint64_t lineStripsPrimIndicesSize = topoInfo->lineStripsNumVertices * sizeof(uint32_t);
	lineStripsInfoCreateInfo.size = lineStripsPrimIndicesSize;
	OP_SmartRef<POP_Buffer>	lineStripsPrimIndicesBuf = myContext->createBuffer(lineStripsInfoCreateInfo, nullptr);
	if (!lineStripsPrimIndicesBuf)
		return;
	uint32_t* lineStripsPrimIndicesData = (uint32_t*)lineStripsPrimIndicesBuf->getData(nullptr);

	// include the restart index
	for (int i = 0; i < lineStrip0Pos.size() + 1; i++)
	{
		*lineStripsPrimIndicesData = 0;
		lineStripsPrimIndicesData++;
	}

	for (int i = 0; i < lineStrip1Pos.size() + 1; i++)
	{
		*lineStripsPrimIndicesData = 1;
		lineStripsPrimIndicesData++;
	}

	// We can't use the buffers once we've given them to this call, so move the reference into this class
	infoBufs.lineStripsInfo = std::move(lineStripsInfoBuf);
	infoBufs.lineStripsPrimIndices = std::move(lineStripsPrimIndicesBuf);

	output->setInfoBuffers(&infoBufs, sinfo, nullptr);
}

// This one uses more attributes and vertex/primitive attributes for the mixedPrimitives
void
PointreceiverPOP::mixedPrimitivesComplex(POP_Output* output)
{
	POP_SetBufferInfo sinfo;

	// to generate a geometry:
	// addPoint() is the first function to be called.
	// then we can add normals, colors, and any custom attributes for the points
	// last function to be called is addLines()

	// Generate some line strips
	// Points 0-8
	std::array<Position, 9> lineStrip0Pos =
	{
		Position(-0.8f, 0.0f, 1.0f),
		Position(-0.6f, 0.4f, 1.0f),
		Position(-0.4f, 0.8f, 1.0f),
		Position(-0.2f, 0.4f, 1.0f),
		Position(0.0f,  0.0f, 1.0f),
		Position(0.2f, -0.4f, 1.0f),
		Position(0.4f, -0.8f, 1.0f),
		Position(0.6f, -0.4f, 1.0f),
		Position(0.8f,  0.0f, 1.0f),
	};

	// Points 9-16
	std::array<Position, 8> lineStrip1Pos =
	{
		Position(-0.8f, 0.2f, 1.0f),
		Position(-0.6f, 0.6f, 1.0f),
		Position(-0.4f, 1.0f, 1.0f),
		Position(-0.2f, 0.6f, 1.0f),
		Position(0.0f,  0.2f, 1.0f),
		Position(0.2f, -0.2f, 1.0f),
		Position(0.4f, -0.6f, 1.0f),
		Position(0.6f, -0.2f, 1.0f),
	};

	// This is a geometry made up of a quad and a triangle, that share some points
	// Points 17-21
	std::array<Position, 5> triAndQuadPos =
	{
		Position(1.0f, 0.0f, 0.0f),
		Position(1.0f, 1.0f, 0.0f),
		Position(0.0f, 1.0f, 0.0f),
		Position(0.0f, 0.0f, 0.0f),
		Position(1.5f, 0.5f, 0.0f),
	};

	POP_BufferInfo attBufCreateInfo;
	uint64_t posSize = getArrayByteSize(lineStrip0Pos) + getArrayByteSize(lineStrip1Pos) + getArrayByteSize(triAndQuadPos);
	attBufCreateInfo.size = posSize;
	OP_SmartRef<POP_Buffer>	posBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!posBuf)
		return;
	Position* posData = (Position*)posBuf->getData(nullptr);

	auto copyArray =
		[&posData](const auto& arr)
		{
			memcpy(posData, arr.data(), getArrayByteSize(arr));
			posData += arr.size();
		};

	copyArray(lineStrip0Pos);
	copyArray(lineStrip1Pos);
	copyArray(triAndQuadPos);

	POP_AttributeInfo posInfo;
	posInfo.numComponents = 3;
	posInfo.type = POP_AttributeType::Float;
	posInfo.name = "P";
	posInfo.attribClass = POP_AttributeClass::Point;
	output->setAttribute(&posBuf, posInfo, sinfo, nullptr);

	////////////////////////////////////////////

	POP_InfoBuffers infoBufs;

	infoBufs.pointInfo = createPointInfoBuffer();
	POP_PointInfo* pointInfo = (POP_PointInfo*)infoBufs.pointInfo->getData(nullptr);
	pointInfo->numPoints = uint32_t(posSize / sizeof(Position));

	infoBufs.topoInfo = createTopologyInfoBuffer();
	POP_TopologyInfo* topoInfo = (POP_TopologyInfo*)infoBufs.topoInfo->getData(nullptr);;

	topoInfo->trianglesStartIndex = 0;
	topoInfo->trianglesCount = 1;

	topoInfo->quadsStartIndex = topoInfo->trianglesStartIndex + topoInfo->trianglesCount * 3;
	topoInfo->quadsCount = 1;

	topoInfo->lineStripsStartIndex = topoInfo->quadsStartIndex + topoInfo->quadsCount * 4;
	// +2 for the restart indices
	topoInfo->lineStripsNumVertices = (uint32_t)(lineStrip0Pos.size() + lineStrip1Pos.size() + 2);
	topoInfo->lineStripsCount = 2;

	////////////////////////////////////////////

	// Per-primitive normals
	std::array<Vector, 4> normal =
	{
		Vector(0.0f, 0.0f, 1.0f),
		Vector(0.0f, 0.0f, 1.0f),
		Vector(0.0f, 0.0f, 1.0f),
		Vector(0.0f, 0.0f, 1.0f),
	};

	uint64_t normalSize = getArrayByteSize(normal);
	attBufCreateInfo.size = normalSize;
	OP_SmartRef<POP_Buffer>	normalBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!normalBuf)
		return;
	memcpy(normalBuf->getData(nullptr), normal.data(), getArrayByteSize(normal));

	POP_AttributeInfo normalInfo;
	normalInfo.numComponents = 3;
	normalInfo.type = POP_AttributeType::Float;
	normalInfo.qualifier = POP_AttributeQualifier::Direction;
	normalInfo.name = "N";
	normalInfo.attribClass = POP_AttributeClass::Primitive;
	output->setAttribute(&normalBuf, normalInfo, sinfo, nullptr);

	////////////////////////////////////////////

	// Per-vertex colors. These need to line up with the entries in the 'indices' buffer below
	std::array<Color, 26> color =
	{
		// triangle
		Color(1.0f, 0.0f, 0.0f, 1.0f),
		Color(0.0f, 1.0f, 0.0f, 1.0f),
		Color(0.0f, 0.0f, 1.0f, 1.0f),

		// quad
		Color(1.0f, 0.0f, 1.0f, 1.0f),
		Color(0.0f, 1.0f, 0.0f, 1.0f),
		Color(0.0f, 0.0f, 1.0f, 1.0f),
		Color(1.0f, 1.0f, 1.0f, 1.0f),

		// lineStrip0
		Color(1.0f, 0.0f, 0.0f, 1.0f),
		Color(1.0f, 0.0f, 0.1f, 1.0f),
		Color(1.0f, 0.0f, 0.2f, 1.0f),
		Color(1.0f, 0.0f, 0.3f, 1.0f),
		Color(1.0f, 0.0f, 0.4f, 1.0f),
		Color(1.0f, 0.0f, 0.5f, 1.0f),
		Color(1.0f, 0.0f, 0.6f, 1.0f),
		Color(1.0f, 0.0f, 0.7f, 1.0f),
		Color(1.0f, 0.0f, 0.8f, 1.0f),
		// Restart index, although not shown, still needs an entry
		Color(1.0f, 1.0f, 1.0f, 1.0f),

		// lineStrip1
		Color(1.0f, 0.8f, 0.0f, 1.0f),
		Color(1.0f, 0.7f, 0.1f, 1.0f),
		Color(1.0f, 0.6f, 0.2f, 1.0f),
		Color(1.0f, 0.5f, 0.3f, 1.0f),
		Color(1.0f, 0.4f, 0.4f, 1.0f),
		Color(1.0f, 0.3f, 0.5f, 1.0f),
		Color(1.0f, 0.2f, 0.6f, 1.0f),
		Color(1.0f, 0.1f, 0.7f, 1.0f),
		// Restart index, although not shown, still needs an entry
		Color(1.0f, 1.0f, 1.0f, 1.0),
	};

	uint64_t colorSize = getArrayByteSize(color);
	attBufCreateInfo.size = colorSize;
	OP_SmartRef<POP_Buffer>	colorBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!colorBuf)
		return;
	memcpy(colorBuf->getData(nullptr), color.data(), getArrayByteSize(color));

	POP_AttributeInfo colorInfo;
	colorInfo.numComponents = 4;
	colorInfo.type = POP_AttributeType::Float;
	colorInfo.name = "Color";
	colorInfo.attribClass = POP_AttributeClass::Vertex;
	output->setAttribute(&colorBuf, colorInfo, sinfo, nullptr);

	////////////////////////////////////////////

	// Notice that altough the different primitive types need to appear in the index buffer in the same
	// order that they appear in the POP_PrimitiveInfo class:
	// triangles, quads, lineStrips, lines, pointPrimitives
	// they can reference point indices anywhere in the buffer.
	std::array<uint32_t, 26> indices =
	{
		// triangle
		18, 17, 21,

		// quad
		17, 18, 19, 20,

		// lineStrip0
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		// line strip restart index
		0xFFFFFFFFU,

		// lineStrip1
		9, 10, 11, 12, 13, 14, 15, 16,
		// line strip restart index
		0xFFFFFFFFU,
	};

	int64_t indexSize = getArrayByteSize(indices);
	POP_BufferInfo indexBufCreateInfo;
	indexBufCreateInfo.usage = POP_BufferUsage::IndexBuffer;
	indexBufCreateInfo.size = indexSize;
	OP_SmartRef<POP_Buffer>	indexBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!indexBuf)
		return;
	memcpy(indexBuf->getData(nullptr), indices.data(), indexSize);

	POP_IndexBufferInfo indexInfo;
	indexInfo.type = POP_IndexType::UInt32;
	output->setIndexBuffer(&indexBuf, indexInfo, sinfo, nullptr);

	////////////////////////////////////////////

	// Each line strip requires two entries in this buffer.
	// One is the start index, the second is the number of points in the strip, including the restart index
	uint64_t lineStripsInfoSize = 2 * 2 * sizeof(uint32_t);

	POP_BufferInfo lineStripsInfoCreateInfo;
	lineStripsInfoCreateInfo.usage = POP_BufferUsage::LineStripsInfoBuffer;
	lineStripsInfoCreateInfo.size = lineStripsInfoSize;

	OP_SmartRef<POP_Buffer>	lineStripsInfoBuf = myContext->createBuffer(lineStripsInfoCreateInfo, nullptr);
	if (!lineStripsInfoBuf)
		return;
	uint32_t* lineStripsInfoData = (uint32_t*)lineStripsInfoBuf->getData(nullptr);
	// First index location in the index buffer of the line strip, starting where the line strips being
	lineStripsInfoData[0] = 0;
	// number of indices in the line strip, including the restart index
	lineStripsInfoData[1] = (uint32_t)(lineStrip0Pos.size() + 1);
	// Don't read from the lineStripsInfoData[1] memory, since that is not allocated as readable.
	lineStripsInfoData[2] = (uint32_t)(lineStrip0Pos.size() + 1);
	lineStripsInfoData[3] = (uint32_t)(lineStrip1Pos.size() + 1);

	////////////////////////////////////////////

	uint64_t lineStripsPrimIndicesSize = topoInfo->lineStripsNumVertices * sizeof(uint32_t);
	lineStripsInfoCreateInfo.size = lineStripsPrimIndicesSize;
	OP_SmartRef<POP_Buffer>	lineStripsPrimIndicesBuf = myContext->createBuffer(lineStripsInfoCreateInfo, nullptr);
	if (!lineStripsPrimIndicesBuf)
		return;

	uint32_t* lineStripsPrimIndicesData = (uint32_t*)lineStripsPrimIndicesBuf->getData(nullptr);
	// include the restart index
	for (int i = 0; i < lineStrip0Pos.size() + 1; i++)
	{
		*lineStripsPrimIndicesData = 0;
		lineStripsPrimIndicesData++;
	}

	for (int i = 0; i < lineStrip1Pos.size() + 1; i++)
	{
		*lineStripsPrimIndicesData = 1;
		lineStripsPrimIndicesData++;
	}

	infoBufs.lineStripsInfo = std::move(lineStripsInfoBuf);
	infoBufs.lineStripsPrimIndices = std::move(lineStripsPrimIndicesBuf);

	output->setInfoBuffers(&infoBufs, sinfo, nullptr);
}

// This one creates a grid and assigns the grid metadata that can be used be some other POPs
void
PointreceiverPOP::gridGeometry(POP_Output* output)
{
	POP_SetBufferInfo sinfo;
	POP_BufferInfo attBufCreateInfo;

	// 4x3 grid
	std::array<uint32_t, 2> gridDims = { 4, 3 };
	uint32_t numPoints = 1;
	for (auto i : gridDims)
		numPoints *= i;
	uint64_t posSize = numPoints * sizeof(Position);
	attBufCreateInfo.size = posSize;
	OP_SmartRef<POP_Buffer>	posBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!posBuf)
		return;
	Position* posData = (Position*)posBuf->getData(nullptr);

	// front
	int cnt = 0;
	posData[cnt++] = Position(0.0f, 0.0f, 0.0f);
	posData[cnt++] = Position(1.0f, 0.0f, 0.0f);
	posData[cnt++] = Position(2.0f, 0.0f, 0.0f);
	posData[cnt++] = Position(3.0f, 0.0f, 0.0f);

	posData[cnt++] = Position(0.0f, 0.0f, 1.0f);
	posData[cnt++] = Position(1.0f, 0.0f, 1.0f);
	posData[cnt++] = Position(2.0f, 0.0f, 1.0f);
	posData[cnt++] = Position(3.0f, 0.0f, 1.0f);

	posData[cnt++] = Position(0.0f, 0.0f, 2.0f);
	posData[cnt++] = Position(1.0f, 0.0f, 2.0f);
	posData[cnt++] = Position(2.0f, 0.0f, 2.0f);
	posData[cnt++] = Position(3.0f, 0.0f, 2.0f);

	POP_AttributeInfo posInfo;
	posInfo.numComponents = 3;
	posInfo.type = POP_AttributeType::Float;
	posInfo.name = "P";
	posInfo.attribClass = POP_AttributeClass::Point;
	output->setAttribute(&posBuf, posInfo, sinfo, nullptr);

	int64_t normalSize = numPoints * sizeof(Vector);
	attBufCreateInfo.size = normalSize;
	OP_SmartRef<POP_Buffer>	normalBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!normalBuf)
		return;
	Vector* normData = (Vector*)normalBuf->getData(nullptr);

	for (uint32_t i = 0; i < numPoints; i++)
		normData[i] = Vector(0.0f, 1.0f, 0.0f);

	POP_AttributeInfo normalInfo;
	normalInfo.numComponents = 3;
	normalInfo.type = POP_AttributeType::Float;
	normalInfo.qualifier = POP_AttributeQualifier::Direction;
	normalInfo.name = "N";
	normalInfo.attribClass = POP_AttributeClass::Point;
	output->setAttribute(&normalBuf, normalInfo, sinfo, nullptr);

	// indices for the quad primitives
	std::array<int32_t, 24> indices =
	{
		// front
		0, 4, 5, 1,
		1, 5, 6, 2,
		2, 6, 7, 3,

		4, 8, 9, 5,
		5, 9, 10, 6,
		6, 10, 11, 7,
	};

	int64_t indexSize = getArrayByteSize(indices);
	POP_BufferInfo indexBufCreateInfo;
	indexBufCreateInfo.usage = POP_BufferUsage::IndexBuffer;
	indexBufCreateInfo.size = indexSize;
	OP_SmartRef<POP_Buffer>	indexBuf = myContext->createBuffer(attBufCreateInfo, nullptr);
	if (!indexBuf)
		return;
	memcpy(indexBuf->getData(nullptr), indices.data(), indexSize);

	POP_IndexBufferInfo indexInfo;
	indexInfo.type = POP_IndexType::UInt32;
	output->setIndexBuffer(&indexBuf, indexInfo, sinfo, nullptr);

	POP_InfoBuffers infoBufs;

	infoBufs.pointInfo = createPointInfoBuffer();
	POP_PointInfo* pointInfo = (POP_PointInfo*)infoBufs.pointInfo->getData(nullptr);
	pointInfo->numPoints = numPoints;

	infoBufs.topoInfo = createTopologyInfoBuffer();
	POP_TopologyInfo* topoInfo = (POP_TopologyInfo*)infoBufs.topoInfo->getData(nullptr);
	topoInfo->quadsCount = (uint32_t)indices.size() / 4;

	infoBufs.gridInfo = createGridInfoBuffer(2);
	POP_GridInfo* gridInfo = (POP_GridInfo*)infoBufs.gridInfo->getData(nullptr);
	gridInfo->gridDimensionsCount = (uint32_t)gridDims.size();
	memcpy(gridInfo->gridDimensions, gridDims.data(), getArrayByteSize(gridDims));

	output->setInfoBuffers(&infoBufs, sinfo, nullptr);
}

static std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>>
getAllAttributes(POP_AttributeClass c, const OP_POPInput* input)
{
	std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>> res;
	for (uint32_t i = 0; i < input->getNumAttributes(c); i++)
	{
		const POP_Attribute* attr = input->getAttribute(c, i, nullptr);
		POP_GetBufferInfo info;
		info.location = POP_BufferLocation::CPU;
		OP_SmartRef<POP_Buffer> buf = attr->getBuffer(info, nullptr);
		res.emplace_back(std::string(attr->info.name), buf);
	}
	return res;
}

static void
copyAllAttributes(POP_AttributeClass c, std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>>& attrs, const OP_POPInput* input, POP_Output* output)
{
	POP_SetBufferInfo sinfo;
	for (auto& p : attrs)
	{
		const std::string& name = p.first;
		const POP_Attribute* attr = input->getAttribute(c, name.c_str(), nullptr);
		if (!attr)
		{
			// This should never happen though, since the attrs array was built up from this input.
			continue;
		}
		Vector* b = (Vector*)p.second->getData(nullptr);
		output->setAttribute(&p.second, attr->info, sinfo, nullptr);
	}
}

void
PointreceiverPOP::execute(POP_Output* output, const OP_Inputs* inputs, void* reserved)
{
	myExecuteCount++;

	// If there is an input connected, we'll just do a copy of the data to the CPU and then back up to the GPU.
	// To show how the download operations work.
	if (inputs->getNumInputs() > 0)
	{
		inputs->enablePar("Shape", false);
		inputs->enablePar("Scale", false);

		const OP_POPInput* input = inputs->getInputPOP(0);

		if (!input)
			return;

		// We issue all the download operations first, then start processing them. This allows multiple
		// downloads to be occuring while we process the data
		std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>> pointAttrs, vertAttrs, primAttrs;
		pointAttrs = getAllAttributes(POP_AttributeClass::Point, input);
		vertAttrs = getAllAttributes(POP_AttributeClass::Vertex, input);
		primAttrs = getAllAttributes(POP_AttributeClass::Primitive, input);

		POP_GetBufferInfo ginfo;
		ginfo.location = POP_BufferLocation::CPU;
		POP_InfoBuffers infoBufs;
		// You can individually get the Info buffers, or just get them all in one call
#if 1
		input->getAllInfoBuffers(&infoBufs, ginfo, nullptr);
#else
		infoBufs.topoInfo = input->getTopologyInfo(ginfo, nullptr);
		infoBufs.pointInfo = input->getPointInfo(ginfo, nullptr);
		infoBufs.lineStripsInfo = input->getLineStripsInfo(ginfo, nullptr);
		infoBufs.lineStripsPrimIndices = input->getLineStripsPrimIndices(ginfo, nullptr);
		infoBufs.gridInfo = input->getGridInfo(ginfo, nullptr);
#endif
		OP_SmartRef<POP_Buffer> indexBuf = input->getIndexBuffer(nullptr)->getBuffer(ginfo, nullptr);

		POP_SetBufferInfo sinfo;

		// How assign the outputs.
		// This example just passes the data though.
		// We can do edits to the buffers as we want though, and set them after editing them.
		copyAllAttributes(POP_AttributeClass::Point, pointAttrs, input, output);
		copyAllAttributes(POP_AttributeClass::Vertex, vertAttrs, input, output);
		copyAllAttributes(POP_AttributeClass::Primitive, primAttrs, input, output);
		output->setIndexBuffer(&indexBuf, input->getIndexBuffer(nullptr)->info, sinfo, nullptr);
		output->setInfoBuffers(&infoBufs, sinfo, nullptr);
	}
	else
	{
		inputs->enablePar("Shape", true);

		int shape = inputs->getParInt("Shape");

		inputs->enablePar("Scale", true);
		double	 scale = inputs->getParDouble("Scale");

		switch (shape)
		{
				// Cube
			case 0:
			{
				cubeGeometry(output, (float)scale);
				break;
			}
			// mixed geometry
			case 1:
			{
				mixedPrimitives(output);
				break;
			}
			// mixed geometry complex
			case 2:
			{
				mixedPrimitivesComplex(output);
				break;
			}
			// grid
			case 3:
			{
				gridGeometry(output);
				break;
			}
			// empty
			case 4:
			{
				// Do nothing
				break;
			}
		}
	}
}

//-----------------------------------------------------------------------------------------------------
//								CHOP, DAT, and custom parameters
//-----------------------------------------------------------------------------------------------------

int32_t
PointreceiverPOP::getNumInfoCHOPChans(void* reserved)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the CHOP. In this example we are just going to send 4 channels.
	return 1;
}

void
PointreceiverPOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void* reserved)
{
	// This function will be called once for each channel we said we'd want to return
	// In this example it'll only be called once.

	if (index == 0)
	{
		chan->name->setString("executeCount");
		chan->value = (float)myExecuteCount;
	}
}

bool
PointreceiverPOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved)
{
	infoSize->rows = 3;
	infoSize->cols = 2;
	// Setting this to false means we'll be assigning values to the table
	// one row at a time. True means we'll do it one column at a time.
	infoSize->byColumn = false;
	return true;
}

void
PointreceiverPOP::getInfoDATEntries(int32_t index,
								int32_t nEntries,
								OP_InfoDATEntries* entries,
								void* reserved)
{
	char tempBuffer[4096];

	if (index == 0)
	{
		// Set the value for the first column
#ifdef _WIN32
		strcpy_s(tempBuffer, "executeCount");
#else // macOS
		strlcpy(tempBuffer, "executeCount", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

		// Set the value for the second column
#ifdef _WIN32
		sprintf_s(tempBuffer, "%d", myExecuteCount);
#else // macOS
		snprintf(tempBuffer, sizeof(tempBuffer), "%d", myExecuteCount);
#endif
		entries->values[1]->setString(tempBuffer);
	}
}

void
PointreceiverPOP::setupParameters(OP_ParameterManager* manager, void* reserved)
{
	// scale
	{
		OP_NumericParameter	np;

		np.name = "Scale";
		np.label = "Scale";
		np.defaultValues[0] = 1.0;
		np.minSliders[0] = -10.0;
		np.maxSliders[0] = 10.0;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// shape
	{
		OP_StringParameter	sp;

		sp.name = "Shape";
		sp.label = "Shape";

		sp.defaultValue = "Cube";

		std::array<const char *, 5> names = { "cube", "mixedprims", "mixedprimscomplex", "grid", "empty" };
		std::array<const char *, 5> labels = { "Cube", "Mixed Primitives", "Mixed Primitives (Complex)", "Grid", "Empty" };

		OP_ParAppendResult res = manager->appendMenu(sp, (int32_t)names.size(), names.data(), labels.data());
		assert(res == OP_ParAppendResult::Success);
	}
}


