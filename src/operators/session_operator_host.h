#pragma once
#include "../structs.h"
#include "../gui/catpuccin.h"
#include "../aabb.h"
#include "operator_host_config.gen.h"
#include <functional>
#include <optional>
#include <vector>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <tbb/parallel_pipeline.h>
#include <tbb/concurrent_queue.h>
#include <thrust/host_vector.h>
#include <thread>
#include <atomic>
#include <pcl/point_cloud.h>

namespace pc::operators {


using Scene3D =
    Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;
using Object3D =
    Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;

class SessionOperatorHost {

public:

	inline static SessionOperatorHost* instance = nullptr;

  static operator_in_out_t run_operators(operator_in_out_t begin,
					 operator_in_out_t end,
					 OperatorHostConfiguration &host_config);

  struct pcl_input_frame {
	  unsigned long timestamp;
	  thrust::host_vector<pc::types::position> positions;
  };
  oneapi::tbb::concurrent_queue<pcl_input_frame> pcl_queue;
  std::atomic<pcl::PointCloud<pcl::PointXYZ>::Ptr> latest_voxels;
  std::atomic<std::shared_ptr<std::vector<pc::AABB>>> latest_clusters;


  SessionOperatorHost(OperatorHostConfiguration &config, Scene3D &scene,
		      Magnum::SceneGraph::DrawableGroup3D &parent_group);

  void draw_imgui_window();

  OperatorHostConfiguration &_config;

  void set_voxel(uid id, pc::types::Float3 position, pc::types::Float3 size = { 0.05f, 0.05f, 0.05f });
  void set_cluster(uid id, pc::types::Float3 position, pc::types::Float3 size = { 0.05f, 0.05f, 0.05f });

private:
  Scene3D &_scene;
  Magnum::SceneGraph::DrawableGroup3D &_parent_group;

  std::jthread _pcl_thread;
  bool _shutdown_pipeline = false;

  void add_operator(OperatorConfigurationVariant operator_config) {
    _config.operators.push_back(operator_config);
  }

  // std::atomic<std::shared_ptr<std::vector<pc::types::position>>> _pcl_data;

  struct pcl_ingest {
	  oneapi::tbb::concurrent_queue<pcl_input_frame>& queue;
	  bool& shutdown_pipeline;
	  pcl_input_frame* operator()(tbb::flow_control& fc) const;
  };

  struct pcl_process {
	  std::atomic<pcl::PointCloud<pcl::PointXYZ>::Ptr>& latest_voxels;
	  std::atomic<std::shared_ptr<std::vector<pc::AABB>>>& latest_clusters;
	  void operator()(pcl_input_frame* frame) const;
  };

};

using OperatorList =
    std::vector<std::reference_wrapper<const SessionOperatorHost>>;

extern operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
			       const OperatorList& operator_list);

extern pc::types::PointCloud apply(const pc::types::PointCloud &point_cloud,
                                   const OperatorList &operator_list);

} // namespace pc::operators
