using System;
using System.Diagnostics;
using System.Runtime;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.VFX;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using System.Linq;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

public class PointReceiver : MonoBehaviour
{

    [DllImport("pointreceiver", EntryPoint = "startNetworkThread", CharSet = CharSet.Ansi)]
    public static extern int StartNetworkThread(string pointCasterAddress);

    [DllImport("pointreceiver", EntryPoint = "stopNetworkThread")]
    public static extern int StopNetworkThread();

    [DllImport("pointreceiver", EntryPoint = "pointCloudAvailable")]
    public static extern bool PointCloudAvailable();

    [DllImport("pointreceiver", EntryPoint = "pointCount")]
    public static extern int GetPointCount();

    [DllImport("pointreceiver", EntryPoint = "pointPositions")]
    public static extern IntPtr GetPointPositionsBuffer();

    [DllImport("pointreceiver", EntryPoint = "pointColors")]
    public static extern IntPtr GetPointColorsBuffer();

    [DllImport("pointreceiver", EntryPoint = "dequeue")]
    public static extern bool DequeuePointCloud();

    public static PointReceiver Instance;

    public string PointCasterAddress = "127.0.0.1:9999";
    public RawImage PositionPreview;
    public RawImage ColorPreview;

    public int PointCount;

    public Color PointTint = new Color(0.5f, 0.5f, 0.5f, 1);
    public float PointSize = 0.05f;

    public ComputeBuffer PositionsBuffer;
    public ComputeBuffer ColorsBuffer;
    public Shader PointShader;
    Material PointMaterial;

    public RenderTexture Positions;
    public RenderTexture Colors;

    ComputeShader TransferShader;
    int TransferKernel;

    void Start()
    {
        TransferShader = Resources.Load<ComputeShader>("PointTransfer");
        TransferKernel = TransferShader.FindKernel("PointTransfer");
        Instance = this;
    }

    void OnEnable() 
    {
        RenderPipelineManager.endCameraRendering += DisposeFrameResources;
        StartNetworkThread(PointCasterAddress);
    }

    void OnDisable() 
    {
        RenderPipelineManager.endCameraRendering -= DisposeFrameResources;
        StopNetworkThread();
    }

    void Update()
    {
        if (DequeuePointCloud())
        {
            PointCount = GetPointCount();
            if (PointCount == 0) return;

            var textureWidth = 512;
            var textureHeight = Mathf.CeilToInt((float) PointCount / 512);

            TransferShader.SetInt("texWidth", textureWidth);

            // Create the output render textures
            if (Positions) Destroy(Positions);
            Positions = new RenderTexture(textureWidth, textureHeight, 0, RenderTextureFormat.ARGBFloat);
            Positions.enableRandomWrite = true;
            Positions.Create();

            TransferShader.SetTexture(TransferKernel, "positions", Positions);

            if (Colors) Destroy(Colors);
            Colors = new RenderTexture(textureWidth, textureHeight, 0, RenderTextureFormat.ARGBFloat);
            Colors.enableRandomWrite = true;
            Colors.Create();

            TransferShader.SetTexture(TransferKernel, "colors", Colors);

            // move the incoming position values into a structured buffer to use on the GPU
            // -- the incoming data comes as a 'position' packed into 64-bits:
            //    16 bits for x, y and z values with another 16-bit padding value
            // -- similar for the colour values, except they're packed into 32-bits with an 
            //    8-bit char per color (the shader also changes colors from bgra to rgba format)

            // var packedPositions = new short[PointCount * 4];
            // var packedColors = new float[PointCount];
            // Marshal.Copy(GetPointPositionsBuffer(), packedPositions, 0, PointCount * 4);
            // Marshal.Copy(GetPointColorsBuffer(), packedColors, 0, PointCount);

	    if (PositionsBuffer != null) PositionsBuffer.Dispose();
	    if (ColorsBuffer != null) ColorsBuffer.Dispose();

	    PositionsBuffer = new ComputeBuffer(PointCount, sizeof(long), ComputeBufferType.Structured);
	    PositionsBuffer.SetData(GetPointPositionsBuffer(), PointCount, sizeof(long));
	    TransferShader.SetBuffer(TransferKernel, "packedPositions", PositionsBuffer);

	    ColorsBuffer = new ComputeBuffer(PointCount, sizeof(float), ComputeBufferType.Structured);
	    ColorsBuffer.SetData(GetPointColorsBuffer(), PointCount, sizeof(int));
	    TransferShader.SetBuffer(TransferKernel, "packedColors", ColorsBuffer);

            // since we unpack them two at a time, we dispatch half as many kernels in the x direction
            int blockX = Mathf.CeilToInt(textureWidth / 8f);
            int blockY = Mathf.CeilToInt(textureHeight / 8f);

            TransferShader.Dispatch(TransferKernel, blockX, blockY, 1);

	    foreach(var vfx in GetComponentsInChildren<VisualEffect>())
	    {
                // vfx.Reinit();
		vfx.SendEvent("Emit");
	    }
        }
    }

    void DisposeFrameResources(ScriptableRenderContext context, Camera camera)
    {
        PositionsBuffer?.Dispose();
        ColorsBuffer?.Dispose();
    }

    // void OnRenderObject()
    // {
    //     if (PointMaterial == null)
    //     {
    //         PointMaterial = new Material(PointShader);
    //         PointMaterial.hideFlags = HideFlags.DontSave;
    //         PointMaterial.EnableKeyword("_COMPUTE_BUFFER");
    //     }

    //     PointCount = GetPointCount();
    //     var positionsPtr = GetPointPositionsBuffer();
    //     var positions = new float[PointCount * 4];
    //     Marshal.Copy(positionsPtr, positions, 0, PointCount * 4);
    //     PositionsBuffer = new ComputeBuffer(PointCount, sizeof(float) * 4, ComputeBufferType.Structured);
    //     PositionsBuffer.SetData(positions);

    //     PointMaterial.SetPass(0);
    //     PointMaterial.SetColor("_Tint", PointTint);
    //     PointMaterial.SetMatrix("_Transform", transform.localToWorldMatrix);
    //     PointMaterial.SetBuffer("_PositionsBuffer", PositionsBuffer);
    //     Graphics.DrawProceduralNow(MeshTopology.Points, PositionsBuffer.count, 1);
        
    //     PositionsBuffer.Dispose();
    // }

    // void Update()
    // {
        // PositionsBuffer?.Dispose();
        // PositionsBuffer = null;
    // }


}
