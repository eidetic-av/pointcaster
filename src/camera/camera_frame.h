#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <pointcaster/core_types.h>
#include <pointcaster_api.h>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace pcl {
template <typename T> class PointCloud;
struct PointXYZRGB;
} // namespace pcl

namespace pc {
class PointCloud;
}

namespace pc::camera {

struct POINTCASTER_API FrameProjectionArgs {
  float fx, fy, cx, cy;
  float pixel_width, pixel_height;
  float4x4 camera_extrinsic;
};

struct CameraFrameData {
  size_t width;
  size_t height;

  static constexpr std::string default_color_key = "color";
  static constexpr std::string default_depth_key = "depth";
  static constexpr std::string default_index_key = "index";

  std::unordered_map<std::string, std::vector<color>> color_buffers;
  std::unordered_map<std::string, std::vector<float>> float_buffers;
  std::unordered_map<std::string, std::vector<int32_t>> int_buffers;

  struct PixelHit {
    int32_t point_index;
    float depth;
  };
  // per-pixel list of all points that projected to each pixel
  std::vector<std::vector<PixelHit>> pixel_hits;

  POINTCASTER_API explicit CameraFrameData(size_t default_buffer_width,
                                           size_t default_buffer_height);

  std::vector<color> &main_color_buffer() {
    return color_buffers[default_color_key];
  }

  std::vector<float> &depth_buffer() {
    return float_buffers[default_depth_key];
  }

  std::vector<int32_t> &index_buffer() {
    return int_buffers[default_index_key];
  }

  // TODO might be better as cpu backend func or free func?
  /// Flood-fill the color buffer into empty pixels...
  POINTCASTER_API void
  flood_fill(int fill_passes,
             const std::string &color_buffer_key = default_color_key,
             const std::string &depth_buffer_key = default_depth_key,
             const std::string &index_buffer_key = default_index_key);

  // TODO better as a cpu backend func or free func rather than member?
  /// Convert to a PCL organized cloud for use with PCL's organized filters.
  POINTCASTER_API std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>
  to_pcl_organized(float fx, float fy, float cx, float cy) const;
};

struct CameraFrame {
  std::string name;
  std::optional<std::shared_ptr<CameraFrameData>> frame_data;
};

using CameraFrameRef = std::reference_wrapper<CameraFrame>;

// TODO all this probably should be in the fringe removal operator too
/// Bit flags matching pcl::OrganizedEdgeBase output labels.

enum EdgeLabelBit : uint32_t {
  EdgeNanBoundary = 1u << 0,
  EdgeOccluding = 1u << 1,
  EdgeOccluded = 1u << 2,
  EdgeHighCurvature = 1u << 3,
  EdgeRgbCanny = 1u << 4,
};

struct FringeMaskOptions {
  /// Which edge label bits seed the mask (OR'd together).
  /// Default: OCCLUDING (foreground side of depth jumps) + NAN_BOUNDARY
  /// (silhouette into empty pixels).
  uint32_t seed_label_bits = EdgeOccluding | EdgeNanBoundary;
  /// Optional gate: when non-zero, the seed is intersected with a dilation of
  /// pixels whose label includes any of these bits. Lets you keep only those
  /// seed pixels that have a corresponding gate label within
  /// `gate_proximity_px` pixels (e.g. OCCLUDING ∩ near-Canny).
  uint32_t gate_label_bits = 0;
  /// Dilation radius applied to the gate before intersection (3x3 cross
  /// passes). Ignored if gate_label_bits == 0.
  int gate_proximity_px = 0;
  /// Number of 3x3 cross dilation passes from the seed into the foreground.
  /// Larger values eat further into the interior of the silhouette.
  int erosion_px = 1;
};

/// Build a width*height row-major 0/255 mask of pixels to discard, seeded from
/// the edge-label image and grown inward by morphological dilation.
POINTCASTER_API std::vector<uint8_t>
compute_fringe_mask(int width, int height, std::span<const uint32_t> labels,
                    const FringeMaskOptions &options);

/// Drop points from `cloud` wherever `drop_mask` is non-zero.
/// Uses pixel_hits for multi-layer removal when available, otherwise falls back
/// to int_buffers["index"] (frontmost only).
/// removal_depth_range <= 0: remove ALL layers at masked pixels.
/// removal_depth_range > 0:  only remove points within this distance (same
///                           units as depth buffer, i.e. mm) behind the
///                           frontmost point at each pixel.
POINTCASTER_API void filter_cloud_by_pixel_mask(
    const CameraFrameData &frame, std::span<const uint8_t> drop_mask,
    pc::PointCloud &cloud, float removal_depth_range = 0.0f);

} // namespace pc::camera