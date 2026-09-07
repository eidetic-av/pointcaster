using System;

// Which point cloud channels a host subscribes to.
//
// With AllChannels ticked the host subscribes with an empty address, which the
// native library treats as every channel pointcaster publishes.
[Serializable]
public class ChannelSubscription
{
    public bool AllChannels = true;
    public string Address = "";

    public string SubscriptionAddress => AllChannels ? "" : Address;
}
