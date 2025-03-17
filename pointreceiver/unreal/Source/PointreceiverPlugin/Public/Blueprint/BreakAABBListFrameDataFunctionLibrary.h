#pragma once

#include "../AABB.h"
#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "LiveLinkTypes.h"

// #include "BreakAABBListFrameDataFunctionLibrary.generated.h"

// UCLASS()
// class POINTRECEIVERPLUGIN_API UBreakAABBListFrameDataFunctionLibrary : public UBlueprintFunctionLibrary
// {
//     GENERATED_BODY()

//    public:
//     UFUNCTION(BlueprintPure, Category = "LiveLink|AABB",
//               meta = (DisplayName = "Break AABBList Frame Data", CompactNodeTitle = "BreakAABB"))
//     static void BreakAABBListFrameData(const FLiveLinkBasicBlueprintData& InFrameData,
//                                        TArray<FPointcasterAABB>& AABBList, float& WorldTime);
// };