#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FPointreceiverRunnable;

class POINTRECEIVERPLUGIN_API FPointreceiverModule : public IModuleInterface
{
   public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    bool TryStartPointreceiverThread();
    void StopPointreceiverThread();

    void SetStreamEnabled(bool bEnabled);
    void HandlePointcasterAddressChanged();

   private:
    void* DllHandle = nullptr;
    bool bEnabled = false;
    TSharedPtr<FPointreceiverRunnable> PointreceiverRunnable;
    FRunnableThread* PointreceiverThread = nullptr;
};
