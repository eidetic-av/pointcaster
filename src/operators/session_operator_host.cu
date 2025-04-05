#include "../logger.h"
#include "../math.h"
#include "denoise/denoise_operator.cuh"
#include "denoise/kdtree.h"
#include "noise_operator.cuh"
#include "pcl/cluster_extraction_operator.gen.h"
#include "rake_operator.cuh"
#include "range_filter_operator.cuh"
#include "rotate_operator.cuh"
#include "sample_filter_operator.cuh"
#include "session_operator_host.h"
#include <algorithm>
#include <deque>
#include <execution>
#include <future>
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

#ifdef WITH_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace pc::operators {

// TODO this function definintion probs shouldnt be in this file
pc::types::PointCloud apply(const pc::types::PointCloud &point_cloud,
                            OperatorList &operator_list) {
  thrust::device_vector<position> positions(point_cloud.size());
  thrust::copy(point_cloud.positions.begin(), point_cloud.positions.end(),
               positions.begin());

  thrust::device_vector<color> colors(point_cloud.size());
  thrust::copy(point_cloud.colors.begin(), point_cloud.colors.end(),
               colors.begin());

  thrust::device_vector<int> indices(point_cloud.size());
  thrust::sequence(indices.begin(), indices.end());

  auto operator_output_begin = thrust::make_zip_iterator(
      thrust::make_tuple(positions.begin(), colors.begin(), indices.begin()));
  auto operator_output_end = thrust::make_zip_iterator(
      thrust::make_tuple(positions.end(), colors.end(), indices.end()));

  for (auto &operator_host_config : operator_list) {
    operator_output_end = pc::operators::SessionOperatorHost::run_operators(
        operator_output_begin, operator_output_end, operator_host_config);
  }

  cudaDeviceSynchronize();

  pc::types::PointCloud result{};

  auto output_point_count =
      std::distance(operator_output_begin, operator_output_end);

  result.positions.resize(output_point_count);
  result.colors.resize(output_point_count);

  thrust::copy(positions.begin(), positions.begin() + output_point_count,
               result.positions.begin());
  thrust::copy(colors.begin(), colors.begin() + output_point_count,
               result.colors.begin());

  return result;
}

// TODO this function definintion probs shouldnt be in this file
operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
                        OperatorList &operator_list) {
  for (auto &operator_host_config : operator_list) {
    end = pc::operators::SessionOperatorHost::run_operators(
        begin, end, operator_host_config);
  }
  return end;
}

operator_in_out_t
SessionOperatorHost::run_operators(operator_in_out_t begin,
                                   operator_in_out_t end,
                                   OperatorHostConfiguration &host_config) {

  for (auto &operator_config : host_config.operators) {
    std::visit(
        [&begin, &end](auto &&config) {
          if (!config.enabled) { return; }

          using T = std::decay_t<decltype(config)>;

#ifdef WITH_TRACY
          ZoneScopedN(T::Name);
#endif

          // Transform operators
          if constexpr (std::is_same_v<T, NoiseOperatorConfiguration>) {
            thrust::transform(begin, end, begin, NoiseOperator{config});
          } else if constexpr (std::is_same_v<T, RotateOperatorConfiguration>) {
            thrust::transform(begin, end, begin, RotateOperator(config));
          } else if constexpr (std::is_same_v<T, RakeOperatorConfiguration>) {
            thrust::transform(begin, end, begin, RakeOperator(config));
          }

          // PCL
          else if constexpr (std::is_same_v<
                                 T, pcl_cpu::ClusterExtractionConfiguration>) {

            using namespace std::chrono;
            using namespace std::chrono_literals;

            auto start_time = system_clock::now();

            // std::thread([positions =
            // thrust::host_vector<pc::types::position>(

            auto &pipeline = pcl_cpu::ClusterExtractionPipeline::instance();
            pipeline.input_queue.emplace(
                pcl_cpu::ClusterExtractionPipeline::InputFrame{
                    .timestamp = 0,
                    .extraction_config = config,
                    .positions = thrust::host_vector<pc::types::position>(
                        thrust::get<0>(begin.get_iterator_tuple()),
                        thrust::get<0>(end.get_iterator_tuple()))});
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
          } else if constexpr (std::is_same_v<
                                   T, RangeFilterOperatorConfiguration>) {

            // TODO: this is RangeFilter Operator behaviour,
            // probs needs to be inside that class somehow

            // the operator host should really not contain logic relating to
            // individual operators, it should simply call them

            // perhaps we completely replace the thrust::copy_if with an
            // operator calling function that returns a new end if the config
            // requires it

            auto starting_point_count = thrust::distance(begin, end);

            thrust::device_vector<position> filtered_positions(
                starting_point_count);
            thrust::device_vector<color> filtered_colors(starting_point_count);
            thrust::device_vector<int> filtered_indices(starting_point_count);

            auto filtered_begin = thrust::make_zip_iterator(thrust::make_tuple(
                filtered_positions.begin(), filtered_colors.begin(),
                filtered_indices.begin()));
            auto filtered_end = thrust::copy_if(begin, end, filtered_begin,
                                                RangeFilterOperator{config});

            int fill_count = thrust::distance(filtered_begin, filtered_end);
            config.fill.fill_count = fill_count;

            if (!config.bypass) {
              end = thrust::copy(filtered_begin, filtered_end, begin);
            }

            if (fill_count > config.fill.count_threshold) {

              const auto fill_value =
                  fill_count / static_cast<float>(config.fill.max_fill);
              const auto fill_proportion =
                  fill_count / static_cast<float>(starting_point_count);

              config.fill.fill_value = fill_value;
              config.fill.proportion = fill_proportion;

              auto minmax_x_points = thrust::minmax_element(
                  filtered_begin, filtered_end, MinMaxXComparator{});
              auto minmax_y_points = thrust::minmax_element(
                  filtered_begin, filtered_end, MinMaxYComparator{});
              auto minmax_z_points = thrust::minmax_element(
                  filtered_begin, filtered_end, MinMaxZComparator{});

              const auto &min_x =
                  thrust::get<0>(*minmax_x_points.first).operator position().x /
                  1000.0f;
              const auto &max_x = thrust::get<0>(*minmax_x_points.second)
                                      .operator position()
                                      .x /
                                  1000.0f;
              const auto &min_y =
                  thrust::get<0>(*minmax_y_points.first).operator position().y /
                  1000.0f;
              const auto &max_y = thrust::get<0>(*minmax_y_points.second)
                                      .operator position()
                                      .y /
                                  1000.0f;
              const auto &min_z =
                  thrust::get<0>(*minmax_z_points.first).operator position().z /
                  1000.0f;
              const auto &max_z = thrust::get<0>(*minmax_z_points.second)
                                      .operator position()
                                      .z /
                                  1000.0f;

              const float box_min_x =
                  config.transform.position.x - config.transform.size.x;
              const float box_max_x =
                  config.transform.position.x + config.transform.size.x;
              const float box_min_y =
                  config.transform.position.y - config.transform.size.y;
              const float box_max_y =
                  config.transform.position.y + config.transform.size.y;
              const float box_min_z =
                  config.transform.position.z - config.transform.size.z;
              const float box_max_z =
                  config.transform.position.z + config.transform.size.z;

              const float min_x_value = pc::math::remap(
                  box_min_x, box_max_x, 0.0f, 1.0f, min_x, true);
              const float max_x_value = pc::math::remap(
                  box_min_x, box_max_x, 0.0f, 1.0f, max_x, true);
              const float min_y_value = pc::math::remap(
                  box_min_y, box_max_y, 0.0f, 1.0f, min_y, true);
              const float max_y_value = pc::math::remap(
                  box_min_y, box_max_y, 0.0f, 1.0f, max_y, true);
              const float min_z_value = pc::math::remap(
                  box_min_z, box_max_z, 0.0f, 1.0f, min_z, true);
              const float max_z_value = pc::math::remap(
                  box_min_z, box_max_z, 0.0f, 1.0f, max_z, true);

              config.minmax.min_x = min_x_value;
              config.minmax.max_x = max_x_value;
              config.minmax.min_y = min_y_value;
              config.minmax.max_y = max_y_value;
              config.minmax.min_z = min_z_value;
              config.minmax.max_z = max_z_value;

              // for a very basic 'energy' parameter, we can calculate the
              // volume and then track changes in the volume over time
              constexpr static auto calculate_volume = [](const auto &mm) {
                return (mm.max_x - mm.min_x) * (mm.max_y - mm.min_y) *
                       (mm.max_z - mm.min_z);
              };

              // // and we keep a cache of the last volume per range operator
              // // instance to be able to calculate the volume change over time
              // using uid = unsigned long int;
              // static std::unordered_map<uid, std::tuple<float, float>>
              // 	  volume_cache; // returns the last time and last volume
              // value

              // using namespace std::chrono;
              // static const auto start_time = duration_cast<milliseconds>(
              // 	  high_resolution_clock::now().time_since_epoch());
              // const auto now = duration_cast<milliseconds>(
              // 	  high_resolution_clock::now().time_since_epoch());
              // const float elapsed_seconds =
              // 	  (now.count() - start_time.count()) / 1000.0f;

              // if (!volume_cache.contains(config.id)) {
              // 	volume_cache[config.id] = {elapsed_seconds, 0.0f};
              // }

              // const auto [cache_time, last_volume] = volume_cache[config.id];
              // if (elapsed_seconds - cache_time >=
              // 	  config.minmax.volume_change_timespan) {
              // 	const auto volume = calculate_volume(config.minmax);
              // 	config.minmax.volume_change = std::abs(volume -
              // last_volume);
              //   volume_cache[config.id] = {elapsed_seconds, volume};
              // }

            } else {
              config.fill.fill_value = 0;
              config.fill.proportion = 0;
              config.minmax.min_x = 0;
              config.minmax.max_x = 0;
              config.minmax.min_y = 0;
              config.minmax.max_y = 0;
              config.minmax.min_z = 0;
              config.minmax.max_z = 0;
            }

            // if (config.fill.publish) {
            // publisher::publish_all(
            //     "fill_value", std::array<float, 1>{config.fill.fill_value},
            //     {"operator", "range_filter", std::to_string(config.id),
            //      "fill"});
            // publisher::publish_all(
            //     "proportion", std::array<float, 1>{config.fill.proportion},
            //     {"operator", "range_filter", std::to_string(config.id),
            //      "fill"});
            // }

            // if (config.minmax.publish) {
            //   auto &mm = config.minmax;
            //   auto id = std::to_string(config.id);
            //   publisher::publish_all(
            // 			     "min_x", std::array<float, 1>{mm.min_x},
            // 			     {"operator", "range_filter", id,
            // "minmax"});
            //   publisher::publish_all(
            // 			     "max_x", std::array<float, 1>{mm.max_x},
            // 			     {"operator", "range_filter", id,
            // "minmax"});
            //   publisher::publish_all(
            // 			     "min_y", std::array<float, 1>{mm.min_y},
            // 			     {"operator", "range_filter", id,
            // "minmax"});
            //   publisher::publish_all(
            // 			     "max_y", std::array<float, 1>{mm.max_y},
            // 			     {"operator", "range_filter", id,
            // "minmax"});
            //   publisher::publish_all(
            // 			     "min_z", std::array<float, 1>{mm.min_z},
            // 			     {"operator", "range_filter", id,
            // "minmax"});
            //   publisher::publish_all(
            // 			     "max_z", std::array<float, 1>{mm.max_z},
            // 			     {"operator", "range_filter", id,
            // "minmax"});
            // }
          }
        },
        operator_config);
  }

  return end;
};

} // namespace pc::operators
