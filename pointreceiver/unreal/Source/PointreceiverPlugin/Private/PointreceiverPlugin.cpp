#include "PointreceiverPlugin.h"

#include "CoreMinimal.h"
#include "HAL/PlatformProcess.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "ILiveLinkClient.h"
#include "ILiveLinkSource.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Modules/ModuleManager.h"
#include "PointreceiverLiveLinkSource.h"
#include "PointreceiverLiveLinkSourceFactory.h"
#include "PointreceiverLog.h"
#include "PointreceiverRunnable.h"
#include "PointreceiverSettings.h"
#include "pointreceiver.h"

void FPointreceiverModule::StartupModule()
{
    UE_LOG(LogPointreceiver, Log, TEXT("Pointreceiver startup module"));

    // Load the external DLL.
    FString DllPath =
        TEXT("D:\\jm\\wam\\terracotta-p4\\ue_terracotta\\Plugins\\PointreceiverPlugin\\Library\\pointreceiver.dll");
    DllHandle = FPlatformProcess::GetDllHandle(*DllPath);
    if (!DllHandle)
    {
        UE_LOG(LogPointreceiver, Error, TEXT("Failed to load pointreceiver.dll from %s"), *DllPath);
        return;
    }

    // Retrieve user settings.
    const UPointreceiverSettings* Settings = GetDefault<UPointreceiverSettings>();
    bEnabled = Settings->StreamEnabled;

    // If streaming is enabled, start the thread and register the Live Link source.
    if (bEnabled)
    {
        TryStartPointreceiverThread();
    }
}

void FPointreceiverModule::ShutdownModule()
{
    UE_LOG(LogPointreceiver, Log, TEXT("Pointreceiver shutdown module"));

    StopPointreceiverThread();

    // Free the external DLL.
    if (DllHandle)
    {
        FPlatformProcess::FreeDllHandle(DllHandle);
        DllHandle = nullptr;
    }
}

bool FPointreceiverModule::TryStartPointreceiverThread()
{
    FIPv4Address Ip;
    const UPointreceiverSettings* Settings = GetDefault<UPointreceiverSettings>();
    bool bValidIp = FIPv4Address::Parse(Settings->PointcasterAddress, Ip);
    if (bValidIp)
    {
        UE_LOG(LogPointreceiver, Log, TEXT("Creating Pointreceiver thread"));
        PointreceiverRunnable = MakeShared<FPointreceiverRunnable>();
        PointreceiverThread =
            FRunnableThread::Create(PointreceiverRunnable.Get(), TEXT("FPointreceiverThread"), 0, TPri_Normal);
        bEnabled = true;
    }
    else
    {
        UE_LOG(LogPointreceiver, Warning, TEXT("Invalid Pointcaster Address: %s"), *Settings->PointcasterAddress);
        bEnabled = false;
        if (UPointreceiverSettings* SettingsMutable = GetMutableDefault<UPointreceiverSettings>())
        {
            SettingsMutable->StreamEnabled = false;
            SettingsMutable->SaveConfig();
        }
    }
    return bEnabled;
}

void FPointreceiverModule::StopPointreceiverThread()
{
    if (PointreceiverRunnable.IsValid())
    {
        PointreceiverRunnable->Stop();
        if (PointreceiverThread)
        {
            PointreceiverThread->WaitForCompletion();
            delete PointreceiverThread;
            PointreceiverThread = nullptr;
        }
        PointreceiverRunnable.Reset();
    }
    bEnabled = false;
}

void FPointreceiverModule::SetStreamEnabled(bool bEnable)
{
    if (bEnable == bEnabled)
        return;
    if (bEnable && !bEnabled)
    {
        TryStartPointreceiverThread();
    }
    else if (!bEnable && bEnabled)
    {
        StopPointreceiverThread();
    }
}

void FPointreceiverModule::HandlePointcasterAddressChanged()
{
    StopPointreceiverThread();
    if (bEnabled)
    {
        TryStartPointreceiverThread();
    }
}

IMPLEMENT_MODULE(FPointreceiverModule, PointreceiverPlugin)
