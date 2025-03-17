#pragma once

#include "CoreMinimal.h"
#include "Roles/LiveLinkBasicRole.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "LiveLinkTypes.h"

#include "AABB.generated.h"

USTRUCT(BlueprintType)
struct POINTRECEIVERPLUGIN_API FPointcasterAABB
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "LiveLink")
    FVector Min;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "LiveLink")
    FVector Max;

    FPointcasterAABB() = default;
    FPointcasterAABB(const FVector& InMin, const FVector& InMax) : Min(InMin), Max(InMax) {}

    FVector Center() const { return (Min + Max) * 0.5f; }
    FVector Size() const
    {
        return FVector(FMath::Abs(Max.X - Min.X), FMath::Abs(Max.Y - Min.Y), FMath::Abs(Max.Z - Min.Z));
    }
};

UCLASS()
class POINTRECEIVERPLUGIN_API UPointcasterAABBFunctionLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

   public:
    UFUNCTION(BlueprintCallable, Category = "LiveLink|AABB")
    static FVector GetAABBCenter(const FPointcasterAABB& AABB) { return AABB.Center(); }

    UFUNCTION(BlueprintCallable, Category = "LiveLink|AABB")
    static FVector GetAABBSize(const FPointcasterAABB& AABB) { return AABB.Size(); }
};

USTRUCT(BlueprintType)
struct POINTRECEIVERPLUGIN_API FPointcasterAABBListStaticData : public FLiveLinkBaseStaticData
{
    GENERATED_BODY()
};

USTRUCT(BlueprintType)
struct POINTRECEIVERPLUGIN_API FPointcasterAABBListFrameData : public FLiveLinkBaseFrameData
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "LiveLink")
    TArray<FPointcasterAABB> AABBList;
};

USTRUCT(BlueprintType)
struct POINTRECEIVERPLUGIN_API FPointcasterAABBListBlueprintData : public FLiveLinkBaseBlueprintData
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiveLink")
    FPointcasterAABBListStaticData StaticData;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiveLink")
    FPointcasterAABBListFrameData FrameData;
};

UCLASS(BlueprintType, meta = (DisplayName = "AABBList Role"))
class POINTRECEIVERPLUGIN_API UPointcasterAABBListRole : public ULiveLinkBasicRole
{
    GENERATED_BODY()
   public:
    virtual FText GetDisplayName() const override { return FText::FromString(TEXT("AABBList")); }

    virtual UScriptStruct* GetStaticDataStruct() const override;
    virtual UScriptStruct* GetFrameDataStruct() const override;
    virtual UScriptStruct* GetBlueprintDataStruct() const override;

    virtual bool InitializeBlueprintData(const FLiveLinkSubjectFrameData& InSourceData,
                                         FLiveLinkBlueprintDataStruct& OutBlueprintData) const override;
};
