#pragma once

#include "Containers/Set.h"
#include "Containers/Map.h"
#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/CriticalSection.h"
#include "HAL/RunnableThread.h"
#include "ILiveLinkSource.h"
#include "Misc/Guid.h"
#include "PointreceiverSettings.h"
#include "pointreceiver.h"

class FPointreceiverLiveLinkSource : public ILiveLinkSource, public TSharedFromThis<FPointreceiverLiveLinkSource>
{
   public:
    FPointreceiverLiveLinkSource();
    virtual ~FPointreceiverLiveLinkSource() override;

    virtual void ReceiveClient(ILiveLinkClient* InClient, FGuid InSourceGuid) override;
    virtual bool IsSourceStillValid() const override;
    virtual bool RequestSourceShutdown() override;
    virtual FText GetSourceType() const override;
    virtual FText GetSourceMachineName() const override;
    virtual FText GetSourceStatus() const override;
    
    static void HandleMessage(const pointreceiver_sync_message& Msg);
    static void HandleInactivitySignal();

    static int GetInstanceCount();

   private:
    static TSet<FPointreceiverLiveLinkSource*> Instances;
    static FCriticalSection InstanceSetAccess;

    ILiveLinkClient* Client;
    FGuid SourceGuid;
    FText SourceType;
    FText SourceMachineName;
    FText SourceStatus;

    TSet<FName> LiveLinkSubjects;
    TMap<FString, FTransform> TransformSubjects;

    void HandleMessageInternal(const pointreceiver_sync_message& Msg);
};
