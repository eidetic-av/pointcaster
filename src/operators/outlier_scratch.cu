// #include <pcl/filters/statistical_outlier_removal.h>
// #include <pcl/gpu/octree/octree.hpp>
// #include <pcl/memory.h>
// #include <pcl/point_cloud.h>

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