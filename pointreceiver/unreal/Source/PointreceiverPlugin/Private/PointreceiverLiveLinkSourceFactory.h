#pragma once

#include "CoreMinimal.h"
#include "LiveLinkSourceFactory.h"
#include "PointreceiverLiveLinkSourceFactory.generated.h"

UCLASS()
class POINTRECEIVERPLUGIN_API UPointreceiverLiveLinkSourceFactory : public ULiveLinkSourceFactory
{
    GENERATED_BODY()
   public:
    virtual FText GetSourceDisplayName() const override { return FText::FromString(TEXT("Pointcaster")); }
    virtual FText GetSourceTooltip() const override
    {
        return FText::FromString(
            TEXT("Connects to the Pointreceiver service to unpack frames from a Pointcaster server"));
    }
    virtual EMenuType GetMenuType() const override { return EMenuType::MenuEntry; }

    virtual TSharedPtr<class ILiveLinkSource> CreateSource(const FString& ConnectionString) const override;
};
