#pragma once

#include "../gui/overlay_text.h"
#include "analyser_2d_config.gen.h"
#include <Magnum/GL/Texture.h>
#include <Magnum/Image.h>
#include <Magnum/Magnum.h>
#include <chrono>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <thread>
#include <optional>

namespace pc::analysis {

struct Analyser2DHost {
  virtual ~Analyser2DHost() = default;
  virtual std::string_view host_id() const = 0;
};

class Analyser2D {

public:
  Analyser2D(const Analyser2DHost *host)
      : _host(host), _analysis_thread([this](auto stop_token) {
	  frame_analysis(stop_token);
	}) {}

  ~Analyser2D();

  Analyser2D(const Analyser2D&) = delete;
  Analyser2D& operator=(const Analyser2D&) = delete;
  
  void set_frame_size(Magnum::Vector2i frame_size);
  void dispatch_analysis(Magnum::GL::Texture2D &texture,
                         Analyser2DConfiguration &config);

  Magnum::GL::Texture2D& analysis_frame();

  int analysis_time();
  std::vector<gui::OverlayText> analysis_labels();

private:
  const Analyser2DHost* _host;
  
  std::optional<Magnum::Image2D> _input_image;
  std::optional<pc::analysis::Analyser2DConfiguration> _input_config;
  Magnum::Vector2i _frame_size;

  std::jthread _analysis_thread;
  std::mutex _dispatch_mutex;
  std::condition_variable _dispatch_condition_variable;

  std::unique_ptr<Magnum::GL::Texture2D> _analysis_frame;
  std::mutex _analysis_frame_mutex;
  std::mutex _analysis_frame_buffer_data_mutex;
  Corrade::Containers::Array<uint8_t> _analysis_frame_buffer_data;
  std::atomic_bool _analysis_frame_buffer_updated;

  std::optional<cv::Mat> _previous_analysis_image;

  std::atomic<std::chrono::milliseconds> _analysis_time;

  std::vector<gui::OverlayText> _analysis_labels;
  std::mutex _analysis_labels_mutex;

  cv::Mat setup_input_frame(Magnum::Image2D &input,
			    const pc::analysis::Analyser2DConfiguration &config);

  std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>,
             std::vector<uchar>>
  calculate_optical_flow(const cv::Mat &input_frame_1,
			 const cv::Mat &input_frame_2,
			 const pc::analysis::OpticalFlowConfiguration &config,
			 const bool use_cuda);

  void frame_analysis(std::stop_token stop_token);
};

} // namespace pc::analysis
