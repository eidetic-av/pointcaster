using UnityEditor;
using UnityEngine;

// Draws a ChannelSubscription as an "All Channels" toggle, revealing the
// address field only when it is unticked.
//
[CustomPropertyDrawer(typeof(ChannelSubscription))]
public class ChannelSubscriptionDrawer : PropertyDrawer
{
    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        var line = EditorGUIUtility.singleLineHeight;
        if (property.FindPropertyRelative("AllChannels").boolValue) return line;
        return line * 2 + EditorGUIUtility.standardVerticalSpacing;
    }

    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        var allChannels = property.FindPropertyRelative("AllChannels");

        EditorGUI.BeginProperty(position, label, property);

        var line = new Rect(position.x, position.y, position.width,
                            EditorGUIUtility.singleLineHeight);
        EditorGUI.PropertyField(line, allChannels, label);

        if (!allChannels.boolValue)
        {
            line.y += EditorGUIUtility.singleLineHeight +
                      EditorGUIUtility.standardVerticalSpacing;
            EditorGUI.indentLevel++;
            EditorGUI.PropertyField(line, property.FindPropertyRelative("Address"),
                                    new GUIContent("Address"));
            EditorGUI.indentLevel--;
        }

        EditorGUI.EndProperty();
    }
}
