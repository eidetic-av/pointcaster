#include "VectorList.h"

#include "Roles/LiveLinkBasicRole.h"
#include "LiveLinkTypes.h"

UScriptStruct* UVectorListRole::GetStaticDataStruct() const
{
    return FVectorListStaticData::StaticStruct();
};

UScriptStruct* UVectorListRole::GetFrameDataStruct() const
{
    return FVectorListFrameData::StaticStruct();
};

UScriptStruct* UVectorListRole::GetBlueprintDataStruct() const
{
    return FVectorListBlueprintData::StaticStruct();
};

bool UVectorListRole::InitializeBlueprintData(const FLiveLinkSubjectFrameData& InSourceData,
                                                       FLiveLinkBlueprintDataStruct& OutBlueprintData) const
{
    bool Success = false;
    FVectorListBlueprintData* BlueprintData = OutBlueprintData.Cast<FVectorListBlueprintData>();
    const FVectorListStaticData* StaticData = InSourceData.StaticData.Cast<FVectorListStaticData>();
    const FVectorListFrameData* FrameData = InSourceData.FrameData.Cast<FVectorListFrameData>();
    if (BlueprintData && StaticData && FrameData)
    {
        GetStaticDataStruct()->CopyScriptStruct(&BlueprintData->StaticData, StaticData);
        GetFrameDataStruct()->CopyScriptStruct(&BlueprintData->FrameData, FrameData);
        Success = true;
    }
    return Success;
};