#include "../structs.h"
#include "operator.h"
#include <thrust/functional.h>

// #include <pcl/point_types.h>

namespace pc::operators {

//  struct PointTypesToPCLXYZ : public thrust::unary_function<indexed_point_t, pcl::PointXYZ>  {

//   __host__ __device__ pcl::PointXYZ operator()(indexed_point_t point) const {
//     auto position = thrust::get<0>(point);
//     auto color = thrust::get<1>(point);

//     pcl::PointXYZ pcl_point;
//     pcl_point.x = static_cast<float>(position.x);
//     pcl_point.y = static_cast<float>(position.y);
//     pcl_point.z = static_cast<float>(position.z);

//     return pcl_point;
//   };
// };

// struct PCLToPointTypes
//   : public thrust::unary_function<pcl::PointXYZRGB, indexed_point_t> {

//   __host__ __device__ indexed_point_t operator()(pcl::PointXYZRGB point) const {

//     auto position = thrust::get<0>(point);
//     auto color = thrust::get<1>(point);

//     pcl::PointXYZRGB pcl_point;
//     pcl_point.x = static_cast<float>(position.x);
//     pcl_point.y = static_cast<float>(position.y);
//     pcl_point.z = static_cast<float>(position.z);
//     pcl_point.r = color.r;
//     pcl_point.g = color.g;
//     pcl_point.b = color.b;

//     return pcl_point;

//   };
// };

} // namespace pc::operators
