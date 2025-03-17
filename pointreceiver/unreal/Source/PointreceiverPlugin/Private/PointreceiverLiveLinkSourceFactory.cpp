#include "PointreceiverLiveLinkSourceFactory.h"

#include "ILiveLinkSource.h"
#include "PointreceiverLiveLinkSource.h"

TSharedPtr<ILiveLinkSource> UPointreceiverLiveLinkSourceFactory::CreateSource(const FString& ConnectionString) const
{
    // TODO ConnectionString is edited in Live Link Source UI,
    // and can be used to set up our LiveLinkSource
    return MakeShared<FPointreceiverLiveLinkSource>();
}
