using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;
using IntPtr = System.IntPtr;

static class ComputeBufferExtension
{
    public unsafe static void SetData
      (this ComputeBuffer buffer, IntPtr pointer, int count, int stride)
    {
        // NativeArray view for the unmanaged memory block
        var view =
          NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<byte>
            ((void*)pointer, count * stride, Allocator.None);

        #if ENABLE_UNITY_COLLECTIONS_CHECKS
        var safety = AtomicSafetyHandle.Create();
        NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref view, safety);
        #endif

        buffer.SetData(view);

        #if ENABLE_UNITY_COLLECTIONS_CHECKS
        AtomicSafetyHandle.Release(safety);
        #endif
    }
}
