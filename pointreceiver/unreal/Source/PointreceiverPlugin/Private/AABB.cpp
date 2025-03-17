#include "AABB.h"

#include "Roles/LiveLinkBasicRole.h"
#include "LiveLinkTypes.h"

UScriptStruct* UPointcasterAABBListRole::GetStaticDataStruct() const
{
    return FPointcasterAABBListStaticData::StaticStruct();
};

UScriptStruct* UPointcasterAABBListRole::GetFrameDataStruct() const
{
    return FPointcasterAABBListFrameData::StaticStruct();
};

UScriptStruct* UPointcasterAABBListRole::GetBlueprintDataStruct() const
{
    return FPointcasterAABBListBlueprintData::StaticStruct();
};

bool UPointcasterAABBListRole::InitializeBlueprintData(const FLiveLinkSubjectFrameData& InSourceData,
                                                       FLiveLinkBlueprintDataStruct& OutBlueprintData) const
{
    bool Success = false;
    FPointcasterAABBListBlueprintData* BlueprintData = OutBlueprintData.Cast<FPointcasterAABBListBlueprintData>();
    const FPointcasterAABBListStaticData* StaticData = InSourceData.StaticData.Cast<FPointcasterAABBListStaticData>();
    const FPointcasterAABBListFrameData* FrameData = InSourceData.FrameData.Cast<FPointcasterAABBListFrameData>();
    if (BlueprintData && StaticData && FrameData)
    {
        GetStaticDataStruct()->CopyScriptStruct(&BlueprintData->StaticData, StaticData);
        GetFrameDataStruct()->CopyScriptStruct(&BlueprintData->FrameData, FrameData);
        Success = true;
    }
    return Success;
};