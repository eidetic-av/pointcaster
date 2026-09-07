using UnityEditor;
using UnityEditor.UIElements;
using UnityEngine.UIElements;

// Draws a ChannelSubscription as an "All Channels" toggle, revealing the
// address field only when it is unticked.
[CustomPropertyDrawer(typeof(ChannelSubscription))]
public class ChannelSubscriptionDrawer : PropertyDrawer
{
    public override VisualElement CreatePropertyGUI(SerializedProperty property)
    {
        var allChannels = property.FindPropertyRelative("AllChannels");

        var root = new VisualElement();
        root.Add(new PropertyField(allChannels, property.displayName));

        var address = new PropertyField(property.FindPropertyRelative("Address"), "Address");
        root.Add(address);

        void SyncAddressVisibility() => address.style.display =
            allChannels.boolValue ? DisplayStyle.None : DisplayStyle.Flex;

        SyncAddressVisibility();
        root.TrackPropertyValue(allChannels, _ => SyncAddressVisibility());

        return root;
    }
}
