using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Rendering;
using System.Collections.Generic;

public class PointreceiverHost : MonoBehaviour
{
    public string PointCasterIp = "127.0.0.1";
    public int PointCloudPort = 9992;

    public ChannelSubscription Subscription = new ChannelSubscription();

    public Text StatusText;
    // public TextMeshPro Text;
    // public RawImage PositionPreview;
    // public RawImage ColorPreview;

    [System.Serializable]
    public class PointCloudFrameResult {
        public int PointCount;
        public ComputeShader TransferShader;
        public int TransferKernel;
        public ComputeBuffer PositionsBuffer;
        public ComputeBuffer ColorsBuffer;
        public RenderTexture Positions;
        public RenderTexture Colors;
    };

    // one result per point cloud channel address, keyed by that address
    public Dictionary<string, PointCloudFrameResult> PointClouds =
        new Dictionary<string, PointCloudFrameResult>();

    Pointreceiver _Pointreceiver;
    string _SubscribedAddress;

    void OnEnable() 
    {
        if (_Pointreceiver == null)
        {
            _Pointreceiver = new Pointreceiver($"unity.{System.Environment.MachineName}");
        }
        try
        {
            _Pointreceiver?.StartPointReceiver($"tcp://{PointCasterIp}:{PointCloudPort}");
            _SubscribedAddress = Subscription.SubscriptionAddress;
            _Pointreceiver?.SubscribeToPointCloud(_SubscribedAddress);
            if (StatusText) StatusText.text = "Started point";
        }
        catch
        {
            Debug.Log("Failed to start Receiver threads");
            if (StatusText) StatusText.text = "Failed to start";
        }
        RenderPipelineManager.endCameraRendering += DisposeFrameResources;
    }

    void OnDisable() 
    {
        try
        {
            if (_SubscribedAddress != null)
            {
                _Pointreceiver?.UnsubscribeFromPointCloud(_SubscribedAddress);
                _SubscribedAddress = null;
            }
            _Pointreceiver?.StopPointReceiver();
            if (StatusText) StatusText.text = "Stopped";
        }
        catch
        {
            Debug.Log("Failed to stop Receiver threads");
            if (StatusText) StatusText.text = "Failed to stop";
        }
        RenderPipelineManager.endCameraRendering -= DisposeFrameResources;
    }

    void OnDestroy()
    {
        _Pointreceiver?.Dispose();
        _Pointreceiver = null;
    }

    void Update()
    {
        if (_Pointreceiver == null) return;

        void ProcessPointCloudFrame(PointCloudFrame frame, PointCloudFrameResult result)
        {
            result.PointCount = frame.PointCount;
            if (result.PointCount == 0) return;

            if (!result.TransferShader)
            {
                result.TransferShader = Resources.Load<ComputeShader>("PointTransfer");
                result.TransferKernel = result.TransferShader.FindKernel("PointTransfer");
            }

            var textureWidth = 512;
            var textureHeight = Mathf.CeilToInt((float)result.PointCount / 512);
            result.TransferShader.SetInt("texWidth", textureWidth);

            // Initialise two render textures, one for color and one for positions
            if (result.Positions) Destroy(result.Positions);
            result.Positions = new RenderTexture(textureWidth, textureHeight, 0, RenderTextureFormat.ARGBFloat);
            result.Positions.enableRandomWrite = true;
            result.Positions.Create();
            result.TransferShader.SetTexture(result.TransferKernel, "positions", result.Positions);
            if (result.Colors) Destroy(result.Colors);
            result.Colors = new RenderTexture(textureWidth, textureHeight, 0, RenderTextureFormat.ARGBFloat);
            result.Colors.enableRandomWrite = true;
            result.Colors.Create();
            result.TransferShader.SetTexture(result.TransferKernel, "colors", result.Colors);

            // move the incoming position values into a structured buffer to use on the GPU
            // -- the incoming data comes as a 'position' packed into 64-bits:
            //    16 bits for x, y and z values with another 16-bit of padding
            // -- the colour values are packed into 32-bits with an 8-bit char per color 
            // -- the shader also changes colors from bgra to rgba format
            // the frame's buffers are borrowed from the native context and die
            // on the next dequeue, so this copy has to happen before we ask for
            // another frame

            var packedPositions = new short[result.PointCount * 4];
            var packedColors = new float[result.PointCount];
            Marshal.Copy(frame.positions, packedPositions, 0, result.PointCount * 4);
            Marshal.Copy(frame.colours, packedColors, 0, result.PointCount);

            result.PositionsBuffer = new ComputeBuffer(result.PointCount * 2, sizeof(int), ComputeBufferType.Structured);
            result.PositionsBuffer.SetData(packedPositions);
            result.TransferShader.SetBuffer(result.TransferKernel, "packedPositions", result.PositionsBuffer);

            result.ColorsBuffer = new ComputeBuffer(result.PointCount, sizeof(float), ComputeBufferType.Structured);
            result.ColorsBuffer.SetData(packedColors);
            result.TransferShader.SetBuffer(result.TransferKernel, "packedColors", result.ColorsBuffer);

            // since we unpack them two at a time, we dispatch half as many kernels in the x direction
            int blockX = Mathf.CeilToInt(textureWidth / 8f);
            int blockY = Mathf.CeilToInt(textureHeight / 8f);
            result.TransferShader.Dispatch(result.TransferKernel, blockX, blockY, 1);
        }

        // the receiver keeps only the newest frame per channel, so draining
        // gives us at most one frame for each of them
        while (_Pointreceiver.TryDequeuePointCloud(1, out string address, out PointCloudFrame incomingFrame))
        {
            if (!PointClouds.TryGetValue(address, out var channelResult))
            {
                channelResult = new PointCloudFrameResult();
                PointClouds.Add(address, channelResult);
            }
            ProcessPointCloudFrame(incomingFrame, channelResult);
        }

        if (StatusText) StatusText.text = $"{PointClouds.Count} channels";
    }

    void DisposeFrameResources(ScriptableRenderContext context, Camera camera)
    {
        foreach (var cloud in PointClouds.Values)
        {
            cloud.PositionsBuffer?.Dispose();
            cloud.ColorsBuffer?.Dispose();
        }
    }

}
