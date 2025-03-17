#include "Blueprint/BreakAABBListFrameDataFunctionLibrary.h"

#include "AABB.h"
#include "LiveLinkTypes.h"
#include "Roles/LiveLinkBasicTypes.h"

// void UBreakAABBListFrameDataFunctionLibrary::BreakAABBListFrameData(const FLiveLinkBasicBlueprintData& InFrameData,
//                                                                     TArray<FPointcasterAABB>& AABBList,
//                                                                     float& WorldTime)
// {
    // const FPointcasterAABBListFrameData* CustomData =
    //     static_cast<const FPointcasterAABBListFrameData*>(&InFrameData.FrameData);

    // if (CustomData)
    // {
    //     AABBList = CustomData->AABBList;
    //     WorldTime = CustomData->WorldTime.GetOffsettedTime();
    // }
    // else
    // {
    //     UE_LOG(LogTemp, Warning,
    //            TEXT("BreakAABBListFrameData: Failed to cast FrameData to FPointcasterAABBListFrameData"));
    // }
// }
