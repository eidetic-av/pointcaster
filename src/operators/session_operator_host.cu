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
              end = thrust::copy_if(begin, end, begin,
                                    RangeFilterOperator{config});
              fill_count = std::distance(begin, end);
            } else {
              fill_count =
                  thrust::count_if(begin, end, RangeFilterOperator{config});
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
