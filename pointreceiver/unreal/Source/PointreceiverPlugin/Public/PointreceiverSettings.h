#pragma once

#include "Engine/DeveloperSettings.h"
#include "PointreceiverSettings.generated.h"

UCLASS(config = PointreceiverPlugin, defaultconfig, meta = (DisplayName = "Pointreceiver"))
class POINTRECEIVERPLUGIN_API UPointreceiverSettings : public UDeveloperSettings
{
    GENERATED_BODY()

   public:
    UPointreceiverSettings();

    UPROPERTY(Config, EditAnywhere, Category = "Stream Settings")
    bool StreamEnabled = false;

    UPROPERTY(Config, EditAnywhere, Category = "Stream Settings")
    FString PointcasterAddress = TEXT("127.0.0.1");

    UPROPERTY(Config, EditAnywhere, Category = "Stream Settings", meta = (ClampMin = "0", UIMin = "0"))
    int32 DequeueTimeoutMs = 50;

#if WITH_EDITOR
    virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
};