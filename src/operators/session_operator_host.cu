#include "../logger.h"
#include "../publisher/publisher.h"
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
          } else if constexpr (std::is_same_v<
                                   T, RangeFilterOperatorConfiguration>) {

	    auto starting_point_count = std::distance(begin, end);
            auto fill_count = 0;

            if (!config.bypass) {
              end = thrust::copy_if(begin, end, begin, RangeFilterOperator{config});
              fill_count = std::distance(begin, end);
            } else {
              fill_count = thrust::count_if(begin, end, RangeFilterOperator{config});
            }

            // TODO: this is RangeFilter Operator behaviour,
	    // probs needs to be inside that class somehow
            config.fill.fill_value =
                fill_count / static_cast<float>(config.fill.max_fill);
	    config.fill.proportion = fill_count / static_cast<float>(starting_point_count);

            if (config.fill.publish) {
              publisher::publish_all(
		  "fill_value", std::array<float, 1>{config.fill.fill_value},
		  {"operator", "range_filter", std::to_string(config.id),
		   "fill"});
              publisher::publish_all(
		  "proportion", std::array<float, 1>{config.fill.proportion},
		  {"operator", "range_filter", std::to_string(config.id),
		   "fill"});
            }

	    auto minmax_x_points =
		thrust::minmax_element(begin, end, MinMaxXComparator{});
	    auto minmax_y_points =
		thrust::minmax_element(begin, end, MinMaxXComparator{});
	    auto minmax_z_points =
		thrust::minmax_element(begin, end, MinMaxXComparator{});

	    const auto &min_x =
		thrust::get<0>(*minmax_x_points.first).operator position().x /
		1000.0f;
	    const auto &max_x =
		thrust::get<0>(*minmax_x_points.second).operator position().x /
		1000.0f;
	    const auto &min_y =
		thrust::get<0>(*minmax_y_points.first).operator position().y /
		1000.0f;
	    const auto &max_y =
		thrust::get<0>(*minmax_y_points.second).operator position().y /
		1000.0f;
	    const auto &min_z =
		thrust::get<0>(*minmax_z_points.first).operator position().z /
		1000.0f;
	    const auto &max_z =
		thrust::get<0>(*minmax_z_points.second).operator position().z /
		1000.0f;

	    float box_min_x = config.position.x - config.size.x;
	    float box_max_x = config.position.x + config.size.x;
	    float box_min_y = config.position.y - config.size.y;
	    float box_max_y = config.position.y + config.size.y;
	    float box_min_z = config.position.z - config.size.z;
	    float box_max_z = config.position.z + config.size.z;

            float min_box_x = (min_x - box_min_x) / (box_max_x - box_min_x);
            float max_box_x = (max_x - box_min_x) / (box_max_x - box_min_x);
            float min_box_y = (min_y - box_min_y) / (box_max_y - box_min_y);
            float max_box_y = (max_y - box_min_y) / (box_max_y - box_min_y);
            float min_box_z = (min_z - box_min_z) / (box_max_z - box_min_z);
            float max_box_z = (max_z - box_min_z) / (box_max_z - box_min_z);

            config.minmax.min_x = min_box_x;
            config.minmax.max_x = max_box_x;
            config.minmax.min_y = min_box_y;
            config.minmax.max_y = max_box_y;
            config.minmax.min_z = min_box_z;
            config.minmax.max_z = max_box_z;

            if (config.minmax.publish) {
	      auto& mm = config.minmax;
	      auto id = std::to_string(config.id);
	      publisher::publish_all(
		  "min_x", std::array<float, 1>{mm.min_x},
		  {"operator", "range_filter", id, "minmax"});
	      publisher::publish_all(
		  "max_x", std::array<float, 1>{mm.max_x},
		  {"operator", "range_filter", id, "minmax"});
	      publisher::publish_all(
		  "min_y", std::array<float, 1>{mm.min_y},
		  {"operator", "range_filter", id, "minmax"});
	      publisher::publish_all(
		  "max_y", std::array<float, 1>{mm.max_y},
		  {"operator", "range_filter", id, "minmax"});
	      publisher::publish_all(
		  "min_z", std::array<float, 1>{mm.min_z},
		  {"operator", "range_filter", id, "minmax"});
	      publisher::publish_all(
		  "max_z", std::array<float, 1>{mm.max_z},
		  {"operator", "range_filter", id, "minmax"});
            }
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
}

} // namespace pc::operators
