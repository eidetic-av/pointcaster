using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;

public class PointreceiverMeshHost : MonoBehaviour
{
    public string PointCasterAddress = "tcp://127.0.0.1";
    public int PointCloudPort = 9992;

    public ChannelSubscription Subscription = new ChannelSubscription();

    // instantiated once per incoming channel; needs a MeshFilter and a
    // MeshRenderer
    public GameObject PointCloudTemplate;

    [Serializable]
    public struct PointCloudChannel
    {
        public string Address;
        public GameObject GameObject;
        public Mesh Mesh;
    }

    public List<PointCloudChannel> PointCloudChannels = new List<PointCloudChannel>();

    private Pointreceiver Pointreceiver;
    private string SubscribedAddress;

    private int CurrentCapacity = 0;
    private int[] RawPositions, RawColors, Indices;
    private NativeArray<int> NativePositions;
    private NativeArray<int> NativeColors;
    private NativeArray<Vector3> OutputPositions;
    private NativeArray<Color32> OutputColors;

    void OnEnable()
    {
        try
        {
            if (Pointreceiver == null)
                Pointreceiver = new Pointreceiver($"unity.{Environment.MachineName}");

            Pointreceiver.StartPointReceiver($"{PointCasterAddress}:{PointCloudPort}");
            SubscribedAddress = Subscription.SubscriptionAddress;
            Pointreceiver.SubscribeToPointCloud(SubscribedAddress);
        }
        catch (Exception e)
        {
            Debug.Log("Pointreceiver Mesh Host");
            Debug.LogError(e);
        }
    }

    void OnDisable()
    {
        try
        {
            if (SubscribedAddress != null)
            {
                Pointreceiver.UnsubscribeFromPointCloud(SubscribedAddress);
                SubscribedAddress = null;
            }
            Pointreceiver.StopPointReceiver();
        }
        catch
        {
            Debug.Log("Failed to stop Receiver threads");
        }
    }

    void OnDestroy()
    {
        if (NativePositions.IsCreated) NativePositions.Dispose();
        if (NativeColors.IsCreated) NativeColors.Dispose();
        if (OutputPositions.IsCreated) OutputPositions.Dispose();
        if (OutputColors.IsCreated) OutputColors.Dispose();

        Pointreceiver?.Dispose();
        Pointreceiver = null;
    }

    void Update()
    {
        if (Pointreceiver == null) return;

        // the receiver keeps only the newest frame per channel, so draining
        // gives us at most one frame for each of them. each frame's buffers are
        // borrowed and die on the next dequeue, so unpack before looping.
        while (Pointreceiver.TryDequeuePointCloud(0, out string address, out PointCloudFrame frame))
        {
            if (frame.PointCount == 0) continue;
            var mesh = EnsureOrCreateChannelMesh(address);
            UnpackPointCloudIntoMesh(frame, mesh);
        }
    }

    void EnsureCapacity(int pointCount)
    {
        if (CurrentCapacity >= pointCount) return;

        CurrentCapacity = pointCount;

        if (NativePositions.IsCreated) NativePositions.Dispose();
        if (NativeColors.IsCreated) NativeColors.Dispose();
        if (OutputPositions.IsCreated) OutputPositions.Dispose();
        if (OutputColors.IsCreated) OutputColors.Dispose();

        RawPositions = new int[CurrentCapacity * 2];
        RawColors = new int[CurrentCapacity];
        Indices = new int[CurrentCapacity];
        for (int i = 0; i < CurrentCapacity; i++)
            Indices[i] = i;

        NativePositions = new NativeArray<int>(CurrentCapacity * 2, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        NativeColors = new NativeArray<int>(CurrentCapacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        OutputPositions = new NativeArray<Vector3>(CurrentCapacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        OutputColors = new NativeArray<Color32>(CurrentCapacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
    }

    void UnpackPointCloudIntoMesh(PointCloudFrame frame, Mesh targetMesh)
    {
        int pointCount = frame.PointCount;
        if (pointCount == 0) return;

        EnsureCapacity(pointCount);

        Marshal.Copy(frame.positions, RawPositions, 0, pointCount * 2);
        Marshal.Copy(frame.colours, RawColors, 0, pointCount);

        NativePositions.CopyFrom(RawPositions);
        NativeColors.CopyFrom(RawColors);

        var unpackJob = new PointUnpackJob
        {
            PackedPosition = NativePositions,
            PackedColor = NativeColors,
            OutPosition = OutputPositions,
            OutColor = OutputColors,
            Count = pointCount
        };
        var handle = unpackJob.Schedule(pointCount, 64);
        handle.Complete();

        ApplyMeshData(targetMesh, pointCount);
    }

    void ApplyMeshData(Mesh mesh, int pointCount)
    {
        mesh.Clear();
        mesh.SetVertexBufferParams(
            pointCount,
            new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3, 0),
            new VertexAttributeDescriptor(VertexAttribute.Color, VertexAttributeFormat.UNorm8, 4, 1)
        );
        mesh.SetVertexBufferData(OutputPositions, 0, 0, pointCount, 0, MeshUpdateFlags.DontRecalculateBounds);
        mesh.SetVertexBufferData(OutputColors, 0, 0, pointCount, 1, MeshUpdateFlags.DontRecalculateBounds);
        mesh.SetIndices(Indices, 0, pointCount, MeshTopology.Points, 0, calculateBounds: false);
        mesh.RecalculateBounds();
    }

    Mesh EnsureOrCreateChannelMesh(string address)
    {
        var existing = PointCloudChannels.FirstOrDefault(channel => channel.Address == address);
        if (existing.Address != null)
            return existing.Mesh;

        var newCloudObject = Instantiate(
            PointCloudTemplate,
            parent: transform,
            worldPositionStays: false
        );
        newCloudObject.name = address;

        var newMesh = new Mesh { indexFormat = IndexFormat.UInt32 };
        newMesh.MarkDynamic();
        newCloudObject.GetComponent<MeshFilter>().sharedMesh = newMesh;

        var meshRenderer = newCloudObject.GetComponent<MeshRenderer>();
        meshRenderer.material = meshRenderer.sharedMaterial;

        PointCloudChannels.Add(new PointCloudChannel
        {
            Address = address,
            GameObject = newCloudObject,
            Mesh = newMesh
        });

        return newMesh;
    }

    [BurstCompile]
    private struct PointUnpackJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int> PackedPosition;
        [ReadOnly] public NativeArray<int> PackedColor;

        public NativeArray<Vector3> OutPosition;
        public NativeArray<Color32> OutColor;

        public int Count;

        public void Execute(int i)
        {
            int posIndex = i * 2;
            int packedXY = PackedPosition[posIndex];
            int packedZ = PackedPosition[posIndex + 1];

            short xi = (short)(packedXY & 0xFFFF);
            short yi = (short)((packedXY >> 16) & 0xFFFF);
            short zi = (short)(packedZ & 0xFFFF);

            float mm = 1000.0f;
            OutPosition[i] = new Vector3(-xi / mm, yi / mm, zi / mm);

            int bgra = PackedColor[i];
            byte b = (byte)(bgra & 0xFF);
            byte g = (byte)((bgra >> 8) & 0xFF);
            byte r = (byte)((bgra >> 16) & 0xFF);
            byte a = (byte)((bgra >> 24) & 0xFF);

            OutColor[i] = new Color32(r, g, b, a);
        }
    }
}
