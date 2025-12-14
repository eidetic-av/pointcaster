#include "../logger.h"
#include "../math.h"
#include "../profiling.h"

#include "denoise/kdtree.h"
#include "denoise/denoise_operator.cuh"
#include "noise_operator.cuh"
#include "operator.h"
#include "operator_host_config.gen.h"
#include "rake_operator.cuh"
#include "range_filter_operator.cuh"
#include "rotate_operator.cuh"
#include "sample_filter_operator.cuh"
#include "session_operator_host.h"
#include "transform_cuda/rgb_gain_operator.cuh"
#include "transform_cuda/translate_operator.cuh"

#include "minmax_extremes.cuh"
#include "pcl/cluster_extraction_operator.gen.h"
#include "transform_cuda/uniform_gain_operator.cuh"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <execution>
#include <future>
#include <pointclouds.h>
#include <thread>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <variant>

// #include <pcl/filters/voxel_grid.h>
// #include <pcl/gpu/containers/device_array.hpp>
// #include <pcl/gpu/octree/octree.hpp>
// #include <pcl/gpu/segmentation/gpu_extract_clusters.h>
// #include <pcl/gpu/segmentation/impl/gpu_extract_clusters.hpp>
// #include <pcl/octree/octree_search.h>
// #include <pcl/point_cloud.h>

namespace pc::operators {

using namespace pc::operators::cuda;
using namespace pc::profiling;

// TODO this function definintion probs shouldnt be in this file
pc::types::PointCloud apply(const pc::types::PointCloud &point_cloud,
                            OperatorList &operator_list,
                            const std::string_view session_id) {
  using namespace pc::profiling;
  ProfilingZone zone("pc::operators::apply");

  thrust::device_vector<position> positions(point_cloud.size());
  thrust::device_vector<color> colors(point_cloud.size());

  thrust::device_vector<int> indices(point_cloud.size());
  thrust::sequence(indices.begin(), indices.end());

  {
    ProfilingZone init_zone("Copy data to GPU");

    thrust::copy(point_cloud.positions.begin(), point_cloud.positions.end(),
                 positions.begin());
    thrust::copy(point_cloud.colors.begin(), point_cloud.colors.end(),
                 colors.begin());
  }

  auto operator_output_begin = thrust::make_zip_iterator(
      thrust::make_tuple(positions.begin(), colors.begin(), indices.begin()));
  auto operator_output_end = thrust::make_zip_iterator(
      thrust::make_tuple(positions.end(), colors.end(), indices.end()));

  {
    ProfilingZone run_operators_zone("Run operators");

    for (size_t i = 0; i < operator_list.size(); i++) {
      auto &operator_config = operator_list[i];
      using T = std::decay_t<decltype(operator_config)>;
      ProfilingZone operator_host_zone(std::format("{} Operators", T::Name));
      operator_output_end = pc::operators::SessionOperatorHost::run_operators(
          operator_output_begin, operator_output_end, operator_config,
          session_id);
    }
  }

  {
    ProfilingZone sync_zone("CUDA Device Synchronize");
    cudaDeviceSynchronize();
  }

  pc::types::PointCloud result{};

  {
    ProfilingZone copy_zone("Copy data to CPU");

    auto output_point_count =
        std::distance(operator_output_begin, operator_output_end);

    result.positions.resize(output_point_count);
    result.colors.resize(output_point_count);

    thrust::copy(positions.begin(), positions.begin() + output_point_count,
                 result.positions.begin());
    thrust::copy(colors.begin(), colors.begin() + output_point_count,
                 result.colors.begin());
  }

  return result;
}

// TODO this function definintion probs shouldnt be in this file
operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
                        OperatorList &operator_list,
                        const std::string_view session_id) {
  for (auto &operator_host_config : operator_list) {
    end = pc::operators::SessionOperatorHost::run_operators(
        begin, end, operator_host_config, session_id);
  }
  return end;
}

// TODO this function definintion probs shouldnt be in this file
pc::types::PointCloud
apply(const pc::types::PointCloud &point_cloud,
      std::vector<OperatorConfigurationVariant> &operator_variant_list,
      const std::string_view session_id) {
  using namespace pc::profiling;
  ProfilingZone zone("pc::operators::apply");

  thrust::device_vector<position> positions(point_cloud.size());
  thrust::device_vector<color> colors(point_cloud.size());

  thrust::device_vector<int> indices(point_cloud.size());
  thrust::sequence(indices.begin(), indices.end());

  {
    ProfilingZone init_zone("Copy data to GPU");

    thrust::copy(point_cloud.positions.begin(), point_cloud.positions.end(),
                 positions.begin());
    thrust::copy(point_cloud.colors.begin(), point_cloud.colors.end(),
                 colors.begin());
  }

  auto operator_output_begin = thrust::make_zip_iterator(
      thrust::make_tuple(positions.begin(), colors.begin(), indices.begin()));
  auto operator_output_end = thrust::make_zip_iterator(
      thrust::make_tuple(positions.end(), colors.end(), indices.end()));

  {
    ProfilingZone run_operators_zone("Run operators");
    operator_output_end = pc::operators::SessionOperatorHost::run_operators(
        operator_output_begin, operator_output_end, operator_variant_list,
        session_id);
  }

  {
    ProfilingZone sync_zone("CUDA Device Synchronize");
    cudaDeviceSynchronize();
  }

  pc::types::PointCloud result{};

  {
    ProfilingZone copy_zone("Copy data to CPU");

    auto output_point_count =
        std::distance(operator_output_begin, operator_output_end);

    result.positions.resize(output_point_count);
    result.colors.resize(output_point_count);

    thrust::copy(positions.begin(), positions.begin() + output_point_count,
                 result.positions.begin());
    thrust::copy(colors.begin(), colors.begin() + output_point_count,
                 result.colors.begin());
  }

  return result;
}

// TODO this function definintion probs shouldnt be in this file
operator_in_out_t
apply(operator_in_out_t begin, operator_in_out_t end,
      std::vector<OperatorConfigurationVariant> &operator_variant_list,
      const std::string_view session_id) {
  end = pc::operators::SessionOperatorHost::run_operators(
      begin, end, operator_variant_list, session_id);
  return end;
}

operator_in_out_t SessionOperatorHost::run_operators(
    operator_in_out_t begin, operator_in_out_t end,
    std::vector<OperatorConfigurationVariant> &operator_variant_list,
    const std::string_view session_id) {

  for (auto &operator_config : operator_variant_list) {
    std::visit(
        [&begin, &end, session_id](auto &&config) {
          if (!config.enabled) { return; }

          using T = std::decay_t<decltype(config)>;

          ProfilingZone operator_zone(T::Name);

          // Transform operators

          // TODO these could probs be done without the compile time is_same
          // checks, maybe by embedding the configuration type inside the
          // Operator type, similar to how DeviceBase<> extends IDevice

          if constexpr (std::is_same_v<T, NoiseOperatorConfiguration>) {
            thrust::transform(begin, end, begin, NoiseOperator{config});
          } else if constexpr (std::is_same_v<T, RotateOperatorConfiguration>) {
            thrust::transform(begin, end, begin, RotateOperator(config));
          } else if constexpr (std::is_same_v<T, RakeOperatorConfiguration>) {
            thrust::transform(begin, end, begin, RakeOperator(config));
          } else if constexpr (std::is_same_v<T,
                                              TranslateOperatorConfiguration>) {
            thrust::transform(begin, end, begin, TranslateOperator(config));
          } else if constexpr (std::is_same_v<T,
                                              RGBGainOperatorConfiguration>) {
            thrust::transform(begin, end, begin, RGBGainOperator(config));
          } else if constexpr (std::is_same_v<
                                   T, UniformGainOperatorConfiguration>) {
            thrust::transform(begin, end, begin, UniformGainOperator(config));
          }

          // PCL
          else if constexpr (std::is_same_v<
                                 T, pcl_cpu::ClusterExtractionConfiguration>) {

            using namespace std::chrono;
            using namespace std::chrono_literals;

            auto start_time = system_clock::now();

            // std::thread([positions =
            // thrust::host_vector<pc::types::position>(

            auto &pipeline =
                pcl_cpu::ClusterExtractionPipeline::instance(session_id);

            auto device_positions_begin =
                thrust::get<0>(begin.get_iterator_tuple());
            auto device_positions_end =
                thrust::get<0>(end.get_iterator_tuple());

            pcl_cpu::ClusterExtractionPipeline::InputFrame frame{
                .timestamp = 0,
                .extraction_config = config,
                .positions = thrust::host_vector<pc::types::position>(
                    device_positions_begin, device_positions_end)};

            // if we're not changing the input point cloud for its next
            // destination operator, we can run the entire pipeline async
            if (!config.reduce_to_voxels && !config.reduce_to_clusters) {
              pipeline.input_queue.emplace(frame);
            } else {
              // if we are altering the pointcloud this frame, we reed to
              // run the reduction synchronously
              auto result =
                  pcl_cpu::ClusterExtractionPipeline::run_cluster_extraction(
                      frame, pipeline.current_clusters);

              if (result.voxelised_cloud) {
                if (config.reduce_to_voxels) {
                  thrust::host_vector<position> reduced_positions;
                  auto &voxels = *result.voxelised_cloud;
                  reduced_positions.resize(voxels.points.size());
                  thrust::transform(voxels.points.begin(), voxels.points.end(),
                                    reduced_positions.begin(),
                                    pcl_cpu::PointXYZToPosition{});
                  thrust::copy(reduced_positions.begin(),
                               reduced_positions.end(), device_positions_begin);
                  auto voxel_count =
                      static_cast<std::ptrdiff_t>(reduced_positions.size());
                  end = begin + voxel_count;

                } else if (config.reduce_to_clusters) {
                  exit(1);
                }
              }
            }

            // pc::logger->debug("input new frame");

            // TODO move these to draw function in session_operator_host.cc

            auto end_time = system_clock::now();
            auto duration_us =
                duration_cast<microseconds>(end_time - start_time);
            auto duration_ms = duration_us.count() / 1000.0f;

            // pc::logger->debug("pcl main took {:.2f}ms", duration_ms);

          }

          // Filters
          else if constexpr (std::is_same_v<
                                 T, SampleFilterOperatorConfiguration>) {
            end = thrust::copy_if(begin, end, begin,
                                  SampleFilterOperator(config));
          }

          else if constexpr (std::is_same_v<T,
                                            RangeFilterOperatorConfiguration>) {

            // TODO: this is RangeFilter Operator behaviour,
            // probs needs to be inside that class somehow

            // the operator host should really not contain logic relating to
            // individual operators, it should simply call them

            // perhaps we completely replace the thrust::copy_if with an
            // operator calling function that returns a new end if the config
            // requires it

            auto starting_point_count = thrust::distance(begin, end);

            {
              auto filtered_count = 0;

              if (config.bypass) {
                ProfilingZone compute_zone("count_if");
                filtered_count =
                    thrust::count_if(begin, end, RangeFilterOperator{config});
              } else {
                ProfilingZone compute_zone("copy_if");
                auto new_end = thrust::copy_if(begin, end, begin,
                                               RangeFilterOperator{config});
                filtered_count = thrust::distance(begin, new_end);
                end = new_end;
              }

              {
                ProfilingZone update_zone("update");

                config.fill.fill_count = filtered_count;

                if (filtered_count > config.fill.count_threshold) {
                  ProfilingZone analysis_zone("analysis");
                  const auto fill_value =
                      filtered_count / static_cast<float>(config.fill.max_fill);
                  const auto fill_proportion =
                      filtered_count / static_cast<float>(starting_point_count);

                  config.fill.fill_value = fill_value;
                  config.fill.proportion = fill_proportion;

                  if (config.minmax.enabled) {
                    ProfilingZone minmax_zone("minmax_elements");

                    Extremes extremes = thrust::transform_reduce(
                        begin, begin + filtered_count, AsExtremes{},
                        make_initial_extremes(), MergeExtremes{});

                    constexpr auto to_float3 =
                        [](const position &p) -> pc::types::Float3 {
                      return {static_cast<float>(p.x) / 1000.0f,
                              static_cast<float>(p.y) / 1000.0f,
                              static_cast<float>(p.z) / 1000.0f};
                    };

                    struct box {
                      float3 min;
                      float3 max;
                    };
                    const box bounding_box = {
                        .min = {.x = config.transform.position.x -
                                     (config.transform.size.x / 2),
                                .y = config.transform.position.y -
                                     (config.transform.size.y / 2),
                                .z = config.transform.position.z -
                                     (config.transform.size.z / 2)},
                        .max{.x = config.transform.position.x +
                                  (config.transform.size.x / 2),
                             .y = config.transform.position.y +
                                  (config.transform.size.y / 2),
                             .z = config.transform.position.z +
                                  (config.transform.size.z / 2)}};

                    constexpr auto get_normalised_position =
                        [](const pc::types::Float3 &pos,
                           const box &bounds) -> pc::types::Float3 {
                      const auto mapped_x = pc::math::remap(
                          bounds.min.x, bounds.max.x, 0.0f, 1.0f, pos.x, true);
                      const auto mapped_y = pc::math::remap(
                          bounds.min.y, bounds.max.y, 0.0f, 1.0f, pos.y, true);
                      const auto mapped_z = pc::math::remap(
                          bounds.min.z, bounds.max.z, 0.0f, 1.0f, pos.z, true);
                      return {mapped_x, mapped_y, mapped_z};
                    };

                    {
                      ProfilingZone host_fetch_zone("minmax host fetch");

                      auto min_x_pos = to_float3(extremes.min_x_pos);
                      auto max_x_pos = to_float3(extremes.max_x_pos);
                      auto min_y_pos = to_float3(extremes.min_y_pos);
                      auto max_y_pos = to_float3(extremes.max_y_pos);
                      auto min_z_pos = to_float3(extremes.min_z_pos);
                      auto max_z_pos = to_float3(extremes.max_z_pos);

                      config.minmax_positions.min_x = min_x_pos;
                      config.minmax_positions.max_x = max_x_pos;
                      config.minmax_positions.min_y = min_y_pos;
                      config.minmax_positions.max_y = max_y_pos;
                      config.minmax_positions.min_z = min_z_pos;
                      config.minmax_positions.max_z = max_z_pos;

                      config.minmax_normalised_positions.min_x =
                          get_normalised_position(min_x_pos, bounding_box);
                      config.minmax_normalised_positions.max_x =
                          get_normalised_position(max_x_pos, bounding_box);
                      config.minmax_normalised_positions.min_y =
                          get_normalised_position(min_y_pos, bounding_box);
                      config.minmax_normalised_positions.max_y =
                          get_normalised_position(max_y_pos, bounding_box);
                      config.minmax_normalised_positions.min_z =
                          get_normalised_position(min_z_pos, bounding_box);
                      config.minmax_normalised_positions.max_z =
                          get_normalised_position(max_z_pos, bounding_box);

                      config.minmax.min_x =
                          config.minmax_normalised_positions.min_x.x;
                      config.minmax.max_x =
                          config.minmax_normalised_positions.max_x.x;
                      config.minmax.min_y =
                          config.minmax_normalised_positions.min_y.y;
                      config.minmax.max_y =
                          config.minmax_normalised_positions.max_y.y;
                      config.minmax.min_z =
                          config.minmax_normalised_positions.min_z.z;
                      config.minmax.max_z =
                          config.minmax_normalised_positions.max_z.z;
                    }
                  }

                  //
                  //   // for a very basic 'energy' parameter, we can
                  //   calculate the
                  //   // volume and then track changes in the volume over
                  //   time constexpr static auto calculate_volume =
                  //       [](const auto &mm) {
                  //         return (mm.max_x - mm.min_x) * (mm.max_y -
                  //         mm.min_y) *
                  //                (mm.max_z - mm.min_z);
                  //       };
                  //
                  //   // // and we keep a cache of the last volume per
                  //   range
                  //   // operator
                  //   // // instance to be able to calculate the volume
                  //   change
                  //   // over time using uid = unsigned long int; static
                  //   // std::unordered_map<uid, std::tuple<float, float>>
                  //   // 	  volume_cache; // returns the last time and
                  //   last volume
                  //   // value
                  //
                  //   // using namespace std::chrono;
                  //   // static const auto start_time =
                  //   // duration_cast<milliseconds>(
                  //   // high_resolution_clock::now().time_since_epoch());
                  //   // const auto now = duration_cast<milliseconds>(
                  //   // high_resolution_clock::now().time_since_epoch());
                  //   // const float elapsed_seconds =
                  //   // 	  (now.count() - start_time.count()) / 1000.0f;
                  //
                  //   // if (!volume_cache.contains(config.id)) {
                  //   // 	volume_cache[config.id] = {elapsed_seconds,
                  //   0.0f};
                  //   // }
                  //
                  //   // const auto [cache_time, last_volume] =
                  //   // volume_cache[config.id]; if (elapsed_seconds -
                  //   cache_time
                  //   // >= 	  config.minmax.volume_change_timespan)
                  //   {
                  //   // const auto volume =
                  //   calculate_volume(config.minmax);
                  //   // 	config.minmax.volume_change = std::abs(volume -
                  //   // last_volume);
                  //   //   volume_cache[config.id] = {elapsed_seconds,
                  //   volume};
                  //   // }
                  // }

                } else {
                  config.fill.fill_value = 0;
                  config.fill.proportion = 0;
                  if (config.minmax.reset_on_empty) {
                    config.minmax.min_x = 0;
                    config.minmax.max_x = 0;
                    config.minmax.min_y = 0;
                    config.minmax.max_y = 0;
                    config.minmax.min_z = 0;
                    config.minmax.max_z = 0;
                  }
                }
              }
            }
          }
        },
        operator_config);
  }
  return end;
}

operator_in_out_t SessionOperatorHost::run_operators(
    operator_in_out_t begin, operator_in_out_t end,
    OperatorHostConfiguration &host_config, const std::string_view session_id) {
  end = SessionOperatorHost::run_operators(begin, end, host_config.operators,
                                           session_id);
  return end;
};

} // namespace pc::operators
