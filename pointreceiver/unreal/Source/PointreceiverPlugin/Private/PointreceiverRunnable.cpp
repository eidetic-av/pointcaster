#include "PointreceiverRunnable.h"

#include "HAL/PlatformProcess.h"
#include "PointreceiverLog.h"
#include "PointreceiverLiveLinkSource.h"

FPointreceiverRunnable::FPointreceiverRunnable() : bStopping(false), Thread(nullptr), bConnected(false)
{
}

FPointreceiverRunnable::~FPointreceiverRunnable()
{
    Stop();
}

bool FPointreceiverRunnable::Init()
{
    return true;
}

uint32 FPointreceiverRunnable::Run()
{
    UE_LOG(LogPointreceiver, Log, TEXT("Pointreceiver thread started"));

    // Retrieve settings
    const UPointreceiverSettings* Settings = GetDefault<UPointreceiverSettings>();
    const char* Address = TCHAR_TO_ANSI(*(Settings->PointcasterAddress));
    int32 DequeueTimeoutMs = Settings->DequeueTimeoutMs;

    // Create the pointreceiver context
    PointreceiverContext = pointreceiver_create_context();
    if (PointreceiverContext)
    {
        FString ComputerName = FPlatformProcess::ComputerName();
        FString ClientName = FString::Printf(TEXT("unreal.%s"), *ComputerName);
        std::string clientName = TCHAR_TO_ANSI(*ClientName);
        pointreceiver_set_client_name(PointreceiverContext, clientName.c_str());
    }

    // Start the receiver
    int start_result = pointreceiver_start_message_receiver(PointreceiverContext, Address);
    if (start_result != 0) 
    {
        UE_LOG(LogPointreceiver, Error, TEXT("Failed to start receiver!"));
    }

    pointreceiver_sync_message Msg;

    float InactivityAccumulator = 0.0f;
    static const float SleepInterval = 0.005f;       // 5ms
    static const float InactivityThreshold = 10.0f;  // 10 seconds

    while (!bStopping)
    {
        if (pointreceiver_dequeue_message(PointreceiverContext, &Msg, DequeueTimeoutMs))
        {
            FPointreceiverLiveLinkSource::HandleMessage(Msg);
            pointreceiver_free_sync_message(PointreceiverContext, &Msg);
            InactivityAccumulator = 0.0f;
        }
        else
        {
            FPlatformProcess::Sleep(SleepInterval);
            InactivityAccumulator += SleepInterval;
            if (InactivityAccumulator >= InactivityThreshold)
            {
                UE_LOG(LogPointreceiver, Warning, TEXT("No messages received from Pointcaster"));
                FPointreceiverLiveLinkSource::HandleInactivitySignal();
                InactivityAccumulator = 0.0f;
            }
        }
    }
    // Stop the receiver
    pointreceiver_stop_message_receiver(PointreceiverContext);

    // Clean up the pointreceiver context
    if (PointreceiverContext)
    {
        pointreceiver_destroy_context(PointreceiverContext);
        PointreceiverContext = nullptr;
    }

    UE_LOG(LogPointreceiver, Log, TEXT("Pointreceiver thread stopped"));
    return 0;
}

void FPointreceiverRunnable::Stop()
{
    bStopping = true;
}
