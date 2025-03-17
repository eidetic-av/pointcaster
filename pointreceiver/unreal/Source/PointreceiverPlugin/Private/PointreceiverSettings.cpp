#include "PointreceiverSettings.h"

#include <PointreceiverPlugin.h>

UPointreceiverSettings::UPointreceiverSettings()
{
    // This determines where it appears in Project Settings.
    CategoryName = TEXT("Plugins");
    SectionName = TEXT("Pointreceiver");
}

#if WITH_EDITOR
void UPointreceiverSettings::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
    Super::PostEditChangeProperty(PropertyChangedEvent);

    FName PropertyName =
        (PropertyChangedEvent.Property != nullptr) ? PropertyChangedEvent.Property->GetFName() : NAME_None;

    // if we toggled streaming
    if (PropertyName == GET_MEMBER_NAME_CHECKED(UPointreceiverSettings, StreamEnabled))
    {
        if (FPointreceiverModule* Module = FModuleManager::GetModulePtr<FPointreceiverModule>("PointreceiverPlugin"))
        {
            Module->SetStreamEnabled(StreamEnabled);
        }
    }

    // if we changed the pointcaster host address
    if (PropertyName == GET_MEMBER_NAME_CHECKED(UPointreceiverSettings, PointcasterAddress))
    {
        if (FPointreceiverModule* Module = FModuleManager::GetModulePtr<FPointreceiverModule>("PointreceiverPlugin"))
        {
            Module->HandlePointcasterAddressChanged();
        }
    }
}
#endif