#pragma once

#include "CoreMinimal.h"
#include "Roles/LiveLinkBasicRole.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "LiveLinkTypes.h"

#include "VectorList.generated.h"

USTRUCT(BlueprintType)
struct POINTRECEIVERPLUGIN_API FVectorListStaticData : public FLiveLinkBaseStaticData
{
    GENERATED_BODY()
};

USTRUCT(BlueprintType)
struct POINTRECEIVERPLUGIN_API FVectorListFrameData : public FLiveLinkBaseFrameData
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "LiveLink")
    TArray<FVector> Vectors;
};

USTRUCT(BlueprintType)
struct POINTRECEIVERPLUGIN_API FVectorListBlueprintData : public FLiveLinkBaseBlueprintData
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiveLink")
    FVectorListStaticData StaticData;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiveLink")
    FVectorListFrameData FrameData;
};

UCLASS(BlueprintType, meta = (DisplayName = "VectorList Role"))
class POINTRECEIVERPLUGIN_API UVectorListRole : public ULiveLinkBasicRole
{
    GENERATED_BODY()
   public:
    virtual FText GetDisplayName() const override { return FText::FromString(TEXT("VectorList")); }

    virtual UScriptStruct* GetStaticDataStruct() const override;
    virtual UScriptStruct* GetFrameDataStruct() const override;
    virtual UScriptStruct* GetBlueprintDataStruct() const override;

    virtual bool InitializeBlueprintData(const FLiveLinkSubjectFrameData& InSourceData,
                                         FLiveLinkBlueprintDataStruct& OutBlueprintData) const override;
};
