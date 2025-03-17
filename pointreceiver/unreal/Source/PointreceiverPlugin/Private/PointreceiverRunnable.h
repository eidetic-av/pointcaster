#pragma once

#include "CoreMinimal.h"
#include "ILiveLinkSource.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "Containers/Set.h"
#include "Misc/Guid.h"

#include "pointreceiver.h"
#include "PointreceiverSettings.h"

class FPointreceiverRunnable : public FRunnable,  public TSharedFromThis<FPointreceiverRunnable>
{
public:
    FPointreceiverRunnable();
    virtual ~FPointreceiverRunnable() override;
    virtual bool Init() override;
    virtual uint32 Run() override;
    virtual void Stop() override;

private:
    pointreceiver_context* PointreceiverContext = nullptr;
    FThreadSafeBool bStopping;
    FRunnableThread* Thread;
    bool bConnected;
};
