#include "../logger.h"
#include "../math.h"
#include "denoise/denoise_operator.cuh"
#include "denoise/kdtree.h"
#include "noise_operator.cuh"
#include "rake_operator.cuh"
#include "range_filter_operator.cuh"
#include "rotate_operator.cuh"
#include "sample_filter_operator.cuh"
#include "session_operator_host.h"
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <tracy/Tracy.hpp>
#include <variant>

// #include <pcl/filters/statistical_outlier_removal.h>
// #include <pcl/gpu/octree/octree.hpp>
// #include <pcl/memory.h>
// #include <pcl/point_cloud.h>

namespace pc::operators {

operator_in_out_t
SessionOperatorHost::run_operators(operator_in_out_t begin,
                                   operator_in_out_t end,
                                   OperatorHostConfiguration &host_config) {

  for (auto &operator_config : host_config.operators) {
    std::visit(
	[&begin, &end](auto &&config) {
	  if (!config.enabled) {
            return;
          }

          using T = std::decay_t<decltype(config)>;

          ZoneScopedN(T::Name);

          // Transform operators
          if constexpr (std::is_same_v<T, NoiseOperatorConfiguration>) {
            thrust::transform(begin, end, begin, NoiseOperator{config});
          } else if constexpr (std::is_same_v<T, RotateOperatorConfiguration>) {
            thrust::transform(begin, end, begin, RotateOperator(config));
          } else if constexpr (std::is_same_v<T, RakeOperatorConfiguration>) {
            thrust::transform(begin, end, begin, RakeOperator(config));
          }

          // Filters
          else if constexpr (std::is_same_v<
                                 T, SampleFilterOperatorConfiguration>) {
            end = thrust::copy_if(begin, end, begin,
                                  SampleFilterOperator(config));
          }

	  //

	  // else if constexpr (std::is_same_v<T, DenoiseOperatorConfiguration>) {

	  //   pc::logger->info("Building KDTree");

	  //   // - copy the positions from the operator_in_out_t onto the CPU into a kdNode[]
	  //   // -- first get the positions
	  //   thrust::device_vector<pc::types::position> gpu_positions;
	  //   gpu_positions.reserve(thrust::distance(begin, end));
	  //   thrust::transform(begin, end, gpu_positions.begin(),
	  // 		      get_position{});

	  //   cudaDeviceSynchronize();

          //   // -- copy them onto the CPU
	  //   std::vector<pc::types::position> cpu_positions(
	  // 	gpu_positions.begin(), gpu_positions.end());

          //   // - create our kdNode vector
          //   std::vector<DynaMap::kdNode> kd_nodes(cpu_positions.size());

	  //   // does the copy need to be parallelized?
	  //   for (int i = 0; i < cpu_positions.size(); i++) {
	  //     kd_nodes[i] = {.id = i, .left = nullptr, .right = nullptr};
	  //     kd_nodes[i].x[0] = cpu_positions[i].x / 1000.0f;
	  //     kd_nodes[i].x[1] = cpu_positions[i].y / 1000.0f;
	  //     kd_nodes[i].x[2] = cpu_positions[i].z / 1000.0f;
	  //   }
	  //   if (cpu_positions.size() > 1000) {
          //     pc::logger->info("cpu[1000].x: {}", cpu_positions.at(1000).x);
          //   }

          //   // - create the kdtree
	  //   DynaMap::kdTree tree;
	  //   tree.kdRoot =
	  // 	tree.buildTree(kd_nodes.data(), kd_nodes.size(), 0, MAX_DIM);

          //   pc::logger->info("Successfully built KDTree: {}",
          //                    tree.kdRoot == nullptr ? "false" : "true");

          //   // DenoiseOperator kernel, passing in a reference or pointer to the tree
	    
	  //   // - kernel output still returns a filtered set into begin and end
	  // }
	    //
	  else if constexpr (std::is_same_v<
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
		.
		operator position()
		.x /
		1000.0f;
	      const auto &min_y =
		thrust::get<0>(*minmax_y_points.first).operator position().y /
		1000.0f;
              const auto &max_y = thrust::get<0>(*minmax_y_points.second)
				      .
				      operator position()
				      .y /
				  1000.0f;
	      const auto &min_z =
		  thrust::get<0>(*minmax_z_points.first).operator position().z /
		  1000.0f;
	      const auto &max_z = thrust::get<0>(*minmax_z_points.second)
				      .
				      operator position()
				      .z /
				  1000.0f;

	      const float box_min_x = config.position.x - config.size.x;
	      const float box_max_x = config.position.x + config.size.x;
	      const float box_min_y = config.position.y - config.size.y;
	      const float box_max_y = config.position.y + config.size.y;
	      const float box_min_z = config.position.z - config.size.z;
	      const float box_max_z = config.position.z + config.size.z;

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
	      // 	  volume_cache; // returns the last time and last volume value

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
	      // 	config.minmax.volume_change = std::abs(volume - last_volume);
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

            // TODO these three kernels could probably be fused into one
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
          // else if constexpr (std::is_same_v<
          //                        T, OutlierFilterOperatorConfiguration>) {
          // if (config.enabled) {

          // // Outlier filtering is PCL based
          // auto point_count = thrust::distance(begin, end);
          // thrust::device_vector<pcl::PointXYZ> gpu_points(point_count);
          // thrust::transform(begin, end, gpu_points.begin(),
          // 			PointTypesToPCLXYZ{});

          // thrust::host_vector<pcl::PointXYZ> host_points(gpu_points.begin(),
          // 						     gpu_points.end());

          // // Copy points to GPU
          // pcl::gpu::Octree::PointCloud point_cloud(
          //     host_points.data(), host_points.size());

          // pcl::gpu::Octree gpu_octree;
          // gpu_octree.setCloud(point_cloud);

          // gpu_octree.build();

          // pcl::gpu::PointCloud<pcl::PointXYZRGB> cloud;

          // thrust::host_vector<pcl::PointXYZRGB> pcl_points(
          //     gpu_points.begin(), gpu_points.end());

          // cloud.width = pcl_points.size();
          // cloud.height = 1; // or appropriate height for your data structure
          // cloud.is_dense = false; // or true, depending on your data
          // cloud.points.assign(pcl_points.data(),
          //                     pcl_points.data() + pcl_points.size());

          // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> filter;
          // filter.setInputCloud(cloud.makeShared());
          // filter.setMeanK(10);
          // filter.setStddevMulThresh(1.0f);
          // filter.filter(cloud);

          // }
          // }
        },
        operator_config);
  }

  return end;
};

operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
                        const OperatorList &operator_list) {
  for (auto &operator_host_ref : operator_list) {
    auto &operator_host = operator_host_ref.get();
    end = pc::operators::SessionOperatorHost::run_operators(
        begin, end, operator_host._config);
  }
  return end;
}

pc::types::PointCloud apply(const pc::types::PointCloud &point_cloud,
                            const OperatorList &operator_list) {
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

  for (auto &operator_host_ref : operator_list) {
    operator_output_end = pc::operators::SessionOperatorHost::run_operators(
	operator_output_begin, operator_output_end,
	operator_host_ref.get()._config);
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

} // namespace pc::operators
