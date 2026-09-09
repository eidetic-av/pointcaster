#if POINTRECEIVER_HAS_VFX_GRAPH
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.VFX;
using UnityEngine.VFX.Utility;

[AddComponentMenu("VFX/Property Binders/PointCloudBinder")]
[VFXBinder("PointCloudBinder")]
public class PointCloudBinder : VFXBinderBase
{
    public PointreceiverHost PointreceiverInstance;

    // the point cloud channel this effect draws, matched exactly against the
    // channel addresses pointcaster publishes
    public string ChannelAddress = "";

    [VFXPropertyBinding("UnityEngine.Texture2D"), FormerlySerializedAs("Positions")]
    public ExposedProperty PositionsProperty = "Positions";

    [VFXPropertyBinding("UnityEngine.Texture2D"), FormerlySerializedAs("Colors")]
    public ExposedProperty ColorsProperty = "Colors";

    [VFXPropertyBinding("int"), FormerlySerializedAs("PointCount")]
    public ExposedProperty PointCountProperty = "PointCount";

    public bool EmitOnUpdate = true;


    void OnValidate()
    {
        if (PointreceiverInstance != null) return;
        var go = GameObject.Find("Pointreceiver")
                ?? GameObject.Find("PointreceiverHost")
                ?? GameObject.Find("Pointreceiver Host");
        if (go) PointreceiverInstance = go.GetComponent<PointreceiverHost>();
    }

    public override void UpdateBinding(VisualEffect visualEffect)
    {
        if (PointreceiverInstance == null) return;
        if (!PointreceiverInstance.PointClouds.TryGetValue(ChannelAddress, out var pointCloud)) return;
        if (pointCloud.PointCount <= 1)
        {
            visualEffect.SetInt(PointCountProperty, 0);
            return;
        }
        if (pointCloud.Positions == null) return;
        if (pointCloud.Colors == null) return;
        visualEffect.SetTexture(PositionsProperty, pointCloud.Positions);
        visualEffect.SetTexture(ColorsProperty, pointCloud.Colors);
        visualEffect.SetInt(PointCountProperty, pointCloud.PointCount);
        if (EmitOnUpdate)
        {
            visualEffect.SendEvent("Emit");
        }
    }

    public override bool IsValid(VisualEffect component) =>
        component.HasTexture(PositionsProperty) && component.HasTexture(ColorsProperty);
}
#endif
