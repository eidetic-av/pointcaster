using Unity.Collections;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.VFX;
using UnityEngine.VFX.Utility;

[AddComponentMenu("VFX/Property Binders/PointCloudBinder")]
[VFXBinder("PointCloudBinder")]
public class PointCloudBinder : VFXBinderBase
{
    [VFXPropertyBinding("UnityEngine.Texture2D"), FormerlySerializedAs("Positions")]
    public ExposedProperty PositionsProperty = "Positions";

    [VFXPropertyBinding("UnityEngine.Texture2D"), FormerlySerializedAs("Colors")]
    public ExposedProperty ColorsProperty = "Colors";

    [VFXPropertyBinding("int"), FormerlySerializedAs("PointCount")]
    public ExposedProperty PointCountProperty = "PointCount";

    public override void UpdateBinding(VisualEffect visualEffect)
    {
        if (PointReceiver.Instance.PointCount <= 1) return;
        if (PointReceiver.Instance.Positions == null) return;
        if (PointReceiver.Instance.Colors == null) return;
        visualEffect.SetTexture(PositionsProperty, PointReceiver.Instance.Positions);
        visualEffect.SetTexture(ColorsProperty, PointReceiver.Instance.Colors);
        visualEffect.SetInt(PointCountProperty, PointReceiver.Instance.PointCount);
    }

    public override bool IsValid(VisualEffect component) =>
        component.HasTexture(PositionsProperty) && component.HasTexture(ColorsProperty);
}