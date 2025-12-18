#include "cuda_analyser_2d.h"
#include <Corrade/PluginManager/AbstractManager.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>


namespace pc::analysis {

class CudaAnalyser2D final : public AbstractCudaAnalyser2D {
public:
  explicit CudaAnalyser2D(Corrade::PluginManager::AbstractManager &manager,
                          Corrade::Containers::StringView plugin)
      : AbstractCudaAnalyser2D{manager, plugin} {}

  cv::Mat
  setup_input_frame_cuda(const cv::Mat &rgba_input,
                         const Analyser2DConfiguration &config) override {
    const int input_w = rgba_input.cols;
    const int input_h = rgba_input.rows;

    cv::cuda::GpuMat gpu_input_mat(rgba_input);
    cv::cuda::GpuMat gpu_return_mat;

    // scale to analysis size if different
    if (input_w != config.resolution[0] || input_h != config.resolution[1]) {
      const auto &resolution = config.resolution;
      if (resolution[0] != 0 && resolution[1] != 0) {
        cv::cuda::resize(gpu_input_mat, gpu_return_mat,
                         {resolution[0], resolution[1]}, 0, 0,
                         cv::INTER_LINEAR);
      } else {
        gpu_return_mat = gpu_input_mat;
      }
    } else {
      gpu_return_mat = gpu_input_mat;
    }

    // convert to grayscale
    cv::cuda::cvtColor(gpu_return_mat, gpu_return_mat, cv::COLOR_RGBA2GRAY);

    // threshold
    cv::cuda::threshold(gpu_return_mat, gpu_return_mat,
                        config.binary_threshold[0], config.binary_threshold[1],
                        cv::THRESH_BINARY);

    // blur
    if (config.blur_size > 0) {
      int blur_size = config.blur_size;
      if ((blur_size % 2) == 0) blur_size += 1;
      if (blur_size > 31) blur_size = 31;

      auto filter = cv::cuda::createGaussianFilter(
          gpu_return_mat.type(), gpu_return_mat.type(),
          cv::Size(blur_size, blur_size), 0);
      filter->apply(gpu_return_mat, gpu_return_mat);
    }

    cv::Mat out;
    gpu_return_mat.download(out);
    return out;
  }

  std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>,
             std::vector<uchar>>
  calculate_optical_flow_cuda(const cv::Mat &input_frame_1,
                              const cv::Mat &input_frame_2,
                              const OpticalFlowConfiguration &config) override {
    std::vector<cv::Point2f> feature_point_positions;
    std::vector<cv::Point2f> new_feature_point_positions;
    std::vector<uchar> status;

    cv::cuda::GpuMat gpu_input_frame_1(input_frame_1);
    cv::cuda::GpuMat gpu_input_frame_2(input_frame_2);

    cv::cuda::GpuMat gpu_feature_point_positions;
    cv::cuda::GpuMat gpu_new_feature_point_positions;
    cv::cuda::GpuMat gpu_feature_point_status;

    auto feature_point_detector = cv::cuda::createGoodFeaturesToTrackDetector(
        gpu_input_frame_2.type(), config.feature_point_count,
        config.cuda_feature_detector_quality_cutoff,
        config.feature_point_distance);

    feature_point_detector->detect(gpu_input_frame_2,
                                   gpu_feature_point_positions);

    if (!gpu_feature_point_positions.empty()) {
      auto optical_flow_filter = cv::cuda::SparsePyrLKOpticalFlow::create();
      optical_flow_filter->calc(
          gpu_input_frame_2, gpu_input_frame_1, gpu_feature_point_positions,
          gpu_new_feature_point_positions, gpu_feature_point_status);

      gpu_feature_point_positions.download(feature_point_positions);
      gpu_new_feature_point_positions.download(new_feature_point_positions);
      gpu_feature_point_status.download(status);
    }

    return {std::move(feature_point_positions),
            std::move(new_feature_point_positions), std::move(status)};
  }
};

} // namespace pc::analysis

CORRADE_PLUGIN_REGISTER(CudaAnalyser2D, pc::analysis::CudaAnalyser2D,
                        "net.pointcaster.CudaAnalyser2D/1.0")