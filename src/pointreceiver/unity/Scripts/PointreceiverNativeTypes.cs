using System;
using System.Runtime.InteropServices;

// Mirrors pointreceiver_status in pointreceiver.h
public enum PointreceiverStatus
{
    Ok = 0,
    ErrorInvalidArgument,
    ErrorAlreadyRunning,
    ErrorNotRunning,
    ErrorConnectionFailed,
    ErrorTimeout,
    ErrorDecodeFailed,
    ErrorOutOfRange,
    ErrorOutOfMemory,
    ErrorInternal
}

// A point cloud frame received from pointcaster
//
// Mirrors pointreceiver_point_cloud_frame
//
//  positions: pointreceiver_position_t[], four int16s per point
//  colours:   pointreceiver_color_t[], four bytes per point
[StructLayout(LayoutKind.Sequential)]
public struct PointCloudFrame
{
    public UIntPtr point_count; // size_t
    public IntPtr positions;
    public IntPtr colours;

    public int PointCount => (int)point_count.ToUInt64();
}
