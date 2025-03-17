#include "PointreceiverLiveLinkSource.h"

#include "AABB.h"
#include "VectorList.h"
#include "HAL/CriticalSection.h"
#include "HAL/PlatformProcess.h"
#include "ILiveLinkClient.h"
#include "PointreceiverLog.h"
#include "Roles/LiveLinkTransformRole.h"
#include "Roles/LiveLinkTransformTypes.h"

TSet<FPointreceiverLiveLinkSource*> FPointreceiverLiveLinkSource::Instances;
FCriticalSection FPointreceiverLiveLinkSource::InstanceSetAccess;

FPointreceiverLiveLinkSource::FPointreceiverLiveLinkSource()
    : Client(nullptr), SourceGuid(FGuid::NewGuid()), SourceType(FText::FromString(TEXT("Pointcaster")))
{
    SourceMachineName = FText::FromString(FPlatformProcess::ComputerName());
    SourceStatus = FText::FromString(TEXT("Not Connected"));
    {
        InstanceSetAccess.Lock();
        Instances.Add(this);
        InstanceSetAccess.Unlock();
    }
}

FPointreceiverLiveLinkSource::~FPointreceiverLiveLinkSource()
{
    InstanceSetAccess.Lock();
    Instances.Remove(this);
    InstanceSetAccess.Unlock();
}

// ILiveLinkSource interface
void FPointreceiverLiveLinkSource::ReceiveClient(ILiveLinkClient* InClient, FGuid InSourceGuid)
{
    Client = InClient;
    SourceGuid = InSourceGuid;
    UE_LOG(LogPointreceiver, Warning, TEXT("New client received"));
    // TODO request params here
}

// TODO
bool FPointreceiverLiveLinkSource::IsSourceStillValid() const
{
    UE_LOG(LogPointreceiver, Warning,
           TEXT("Not yet implemented: Unreal requested if Pointreceiver LiveLink source still valid"));
    return true;
}

bool FPointreceiverLiveLinkSource::RequestSourceShutdown()
{
    UE_LOG(LogPointreceiver, Warning, TEXT("Request source shutdown not yet implemented"));
    return true;
}

FText FPointreceiverLiveLinkSource::GetSourceType() const { return SourceType; }

FText FPointreceiverLiveLinkSource::GetSourceMachineName() const { return SourceMachineName; }

FText FPointreceiverLiveLinkSource::GetSourceStatus() const { return SourceStatus; }

void FPointreceiverLiveLinkSource::HandleMessage(const pointreceiver_sync_message& Msg)
{
    InstanceSetAccess.Lock();
    for (FPointreceiverLiveLinkSource* Instance : Instances)
    {
        if (Instance)
        {
            Instance->HandleMessageInternal(Msg);
        }
    }
    InstanceSetAccess.Unlock();
}

void FPointreceiverLiveLinkSource::HandleInactivitySignal()
{
    InstanceSetAccess.Lock();
    for (FPointreceiverLiveLinkSource* Instance : Instances)
    {
        if (Instance)
        {
            static const FText InactiveStatus = FText::FromString(TEXT("Connected"));
            Instance->SourceStatus = FText(InactiveStatus);
        }
    }
    InstanceSetAccess.Unlock();
}

int FPointreceiverLiveLinkSource::GetInstanceCount()
{
    InstanceSetAccess.Lock();
    auto count = Instances.Num();
    InstanceSetAccess.Unlock();
    return count;
}

void FPointreceiverLiveLinkSource::HandleMessageInternal(const pointreceiver_sync_message& Msg)
{
    if (!Client)
    {
        UE_LOG(LogPointreceiver, Warning, TEXT("LiveLink Client is null"));
        return;
    }

    // if we received a message, we're connected
    static FText ConnectedStatus = FText::FromString(TEXT("Connected"));
    SourceStatus = FText(ConnectedStatus);

    // utilities to help parse different message types
    static const auto TryGetTransformSubjectString = [](const FString& InSubjectString) -> FString*
    {
        static const FString TransformPathComponent = TEXT(".transform");
        int32 Index = InSubjectString.Find(TransformPathComponent, ESearchCase::CaseSensitive, ESearchDir::FromStart);
        if (Index != INDEX_NONE)
        {
            // Allocate a new FString containing the base transform address
            return new FString(InSubjectString.Left(Index + TransformPathComponent.Len()));
        }
        return nullptr;
    };

    FName SubjectName = FName(UTF8_TO_TCHAR(Msg.id));
    FString SubjectString = SubjectName.ToString();

    if (!LiveLinkSubjects.Contains(SubjectName))
    {
        LiveLinkSubjects.Add(SubjectName);
        UE_LOG(LogPointreceiver, Log, TEXT("Added new LiveLink subject: %hs"), Msg.id);

        // if we received a subject that's a 'transform' parameter
        if (FString* TransformSubjectString = TryGetTransformSubjectString(SubjectString))
        {
            // add it as a livelink subject with a Transform role
            TransformSubjects.FindOrAdd(*TransformSubjectString);
            FLiveLinkStaticDataStruct StaticData(FLiveLinkTransformStaticData::StaticStruct());
            Client->PushSubjectStaticData_AnyThread({SourceGuid, FName(*TransformSubjectString)},
                                                    ULiveLinkTransformRole::StaticClass(), MoveTemp(StaticData));
        }
        // otherwise handle non-transform 'basic' or 'custom role' parameters
        else
        {
            if (Msg.value_type == POINTRECEIVER_PARAM_VALUE_FLOAT || Msg.value_type == POINTRECEIVER_PARAM_VALUE_INT)
            {
                // for scalar value types, we just create a basic live link role
                FLiveLinkStaticDataStruct StaticData(FLiveLinkBaseStaticData::StaticStruct());
                if (FLiveLinkBaseStaticData* BaseStaticData = StaticData.Cast<FLiveLinkBaseStaticData>())
                {
                    // with a single property ??
                    BaseStaticData->PropertyNames = {TEXT("Value")};
                }
                Client->PushSubjectStaticData_AnyThread({SourceGuid, SubjectName}, ULiveLinkBasicRole::StaticClass(),
                                                        MoveTemp(StaticData));
            }
            else if (Msg.value_type == POINTRECEIVER_PARAM_VALUE_FLOAT3LIST)
            {
                FLiveLinkStaticDataStruct StaticData(FVectorListStaticData::StaticStruct());
                Client->PushSubjectStaticData_AnyThread(
                    {SourceGuid, SubjectName}, UVectorListRole::StaticClass(), MoveTemp(StaticData));
            }
            else if (Msg.value_type == POINTRECEIVER_PARAM_VALUE_AABBLIST)
            {
                FLiveLinkStaticDataStruct StaticData(FPointcasterAABBListStaticData::StaticStruct());
                Client->PushSubjectStaticData_AnyThread({SourceGuid, SubjectName},
                                                        UPointcasterAABBListRole::StaticClass(), MoveTemp(StaticData));
            }
            else
            {
                UE_LOG(LogPointreceiver, Error, TEXT("Received LiveLink msg with unimplemented data type"));
            }
        }
    }

    if (FString* TransformSubjectString = TryGetTransformSubjectString(SubjectString))
    {
        const auto& Value = Msg.value.float3_val;
        if (auto* Transform = TransformSubjects.Find(*TransformSubjectString))
        {
            // axis conversion from Pointcaster is (X, Y, Z) -> (-Z, X, Y)
            if (SubjectString.Contains(TEXT("transform.position")))
            {
                // m to cm
                FVector NewTranslation = FVector(-Value.z, Value.x, Value.y) * 100.0f;
                Transform->SetTranslation(NewTranslation);
            }
            else if (SubjectString.Contains(TEXT("transform.size")))
            {
                FVector NewScale(-Value.z, Value.x, Value.y);
                Transform->SetScale3D(NewScale);
            }
            else if (SubjectString.Contains(TEXT("transform.euler")))
            {
                // this assumes euler values in degrees
                FRotator NewRotator(-Value.z, Value.x, Value.y);
                Transform->SetRotation(NewRotator.Quaternion());
            }

            // Create frame data struct for a transform role.
            FLiveLinkFrameDataStruct FrameData(FLiveLinkTransformFrameData::StaticStruct());
            if (FLiveLinkTransformFrameData* TransformData = FrameData.Cast<FLiveLinkTransformFrameData>())
            {
                TransformData->Transform = *Transform;
                TransformData->WorldTime = FPlatformTime::Seconds();
                Client->PushSubjectFrameData_AnyThread({SourceGuid, FName(*TransformSubjectString)},
                                                       MoveTemp(FrameData));
            }
        }
        else
        {
            UE_LOG(LogPointreceiver, Error, TEXT("Unable to find LiveLink subject for transform address"));
        }
    }
    else
    {
        // handle custom data types
        if (Msg.value_type == POINTRECEIVER_PARAM_VALUE_FLOAT3LIST)
        {
            FLiveLinkFrameDataStruct FrameData(FVectorListFrameData::StaticStruct());
            // FVectorList value type
            if (FVectorListFrameData* VectorListFrameData = FrameData.Cast<FVectorListFrameData>())
            {
                // loop through each incoming element and
                size_t ElementCount = Msg.value.float3_list_val.count;
                VectorListFrameData->Vectors.Reserve(ElementCount);
                for (size_t i = 0; i < ElementCount; i++)
                {
                    const pointreceiver_float3_t& IncomingFloat3 = Msg.value.float3_list_val.data[i];
                    // axis conversion from Pointcaster is (X, Y, Z) -> (-Z, X, Y)
                    VectorListFrameData->Vectors.Emplace(-IncomingFloat3.z, IncomingFloat3.x, IncomingFloat3.y);
                }
                VectorListFrameData->WorldTime = FPlatformTime::Seconds();
                Client->PushSubjectFrameData_AnyThread({SourceGuid, SubjectName}, MoveTemp(FrameData));
            }
        }
        else if (Msg.value_type == POINTRECEIVER_PARAM_VALUE_AABBLIST)
        {
            FLiveLinkFrameDataStruct FrameData(FPointcasterAABBListFrameData::StaticStruct());
            // AABBList value type
            if (FPointcasterAABBListFrameData* AABBListFrameData = FrameData.Cast<FPointcasterAABBListFrameData>())
            {
                // loop through each incoming box and convert our aabb structure into the right format for live link
                size_t BoxCount = Msg.value.aabb_list_val.count;
                AABBListFrameData->AABBList.Reserve(BoxCount);
                for (size_t i = 0; i < BoxCount; i++)
                {
                    const pointreceiver_aabb_t& IncomingAABB = Msg.value.aabb_list_val.data[i];
                    // axis conversion from Pointcaster is (X, Y, Z) -> (-Z, X, Y)
                    AABBListFrameData->AABBList.Emplace(
                        FVector{-IncomingAABB.min[2], IncomingAABB.min[0], IncomingAABB.min[1]},
                        FVector{-IncomingAABB.max[2], IncomingAABB.max[0], IncomingAABB.max[1]});
                }
                AABBListFrameData->WorldTime = FPlatformTime::Seconds();
                Client->PushSubjectFrameData_AnyThread({SourceGuid, SubjectName}, MoveTemp(FrameData));
            }
        }
        // handle basic/scalar data types
        else
        {
            FLiveLinkFrameDataStruct FrameData(FLiveLinkBaseFrameData::StaticStruct());
            if (FLiveLinkBaseFrameData* BasicData = FrameData.Cast<FLiveLinkBaseFrameData>())
            {
                if (Msg.value_type == POINTRECEIVER_PARAM_VALUE_FLOAT)
                {
                    BasicData->PropertyValues = {Msg.value.float_val};
                }
                else if (Msg.value_type == POINTRECEIVER_PARAM_VALUE_INT)
                {
                    BasicData->PropertyValues = {static_cast<float>(Msg.value.int_val)};
                }
                BasicData->WorldTime = FPlatformTime::Seconds();
                Client->PushSubjectFrameData_AnyThread({SourceGuid, SubjectName}, MoveTemp(FrameData));
            }
        }
    }
}