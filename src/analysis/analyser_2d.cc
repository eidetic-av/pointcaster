#include "analyser_2d.h"
#include "../logger.h"
#include "../publisher.h"
#include <Corrade/Containers/Array.h>
#include <Magnum/GL/BufferImage.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/PixelFormat.h>
#include <mapbox/earcut.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

namespace pc::analysis {

Analyser2D::~Analyser2D() {
  _analysis_thread.request_stop();
  _dispatch_condition_variable.notify_one();
}

void Analyser2D::set_frame_size(Magnum::Vector2i frame_size) {
  std::lock_guard lock(_analysis_frame_mutex);
  _analysis_frame = std::make_unique<Magnum::GL::Texture2D>();
  _analysis_frame->setStorage(1, Magnum::GL::TextureFormat::RGBA8, frame_size);
  _frame_size = frame_size;
}

void Analyser2D::dispatch_analysis(Magnum::GL::Texture2D &texture,
                                   Analyser2DConfiguration &config) {
  if (!config.enabled)
    return;
  std::lock_guard lock_dispatch(_dispatch_mutex);
  // move the image onto the CPU for use in our analysis thread
  _input_image =
      texture.image(0, Magnum::Image2D{Magnum::PixelFormat::RGBA8Unorm});
  _input_config = config;
  _dispatch_condition_variable.notify_one();
}

cv::Mat Analyser2D::setup_input_frame(Magnum::Image2D &input,
                                      const Analyser2DConfiguration &config) {
  auto input_frame_size = input.size();
  cv::Mat input_mat(input_frame_size.y(), input_frame_size.x(), CV_8UC4,
                    input.data());
  cv::Mat return_mat;

  if (config.use_cuda) {
    cv::cuda::GpuMat gpu_input_mat(input_mat);
    cv::cuda::GpuMat gpu_return_mat;

    // scale to analysis size if different
    if (input_frame_size.x() != config.resolution[0] ||
        input_frame_size.y() != config.resolution[1]) {
      auto &resolution = config.resolution;
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

    // convert the image to grayscale
    cv::cuda::cvtColor(gpu_return_mat, gpu_return_mat, cv::COLOR_RGBA2GRAY);

    // binary colour thresholding
    cv::cuda::threshold(gpu_return_mat, gpu_return_mat,
                        config.binary_threshold[0], config.binary_threshold[1],
                        cv::THRESH_BINARY);

    // blur the image
    if (config.blur_size > 0) {

      // for cuda's guassian blurring, blur_size must be an odd number
      auto blur_size = config.blur_size;
      if (blur_size % 2 == 0)
        blur_size += 1;
      // and a maximum of 31
      if (blur_size > 31)
        blur_size = 31;

      auto filter = cv::cuda::createGaussianFilter(
          gpu_return_mat.type(), gpu_return_mat.type(),
          cv::Size(blur_size, blur_size), 0);
      filter->apply(gpu_return_mat, gpu_return_mat);
    }

    // perform canny edge detection
    if (config.canny.enabled) {
      auto canny_detector = cv::cuda::createCannyEdgeDetector(
          config.canny.min_threshold, config.canny.max_threshold,
          config.canny.aperture_size);
      canny_detector->detect(gpu_return_mat, gpu_return_mat);
    }

    gpu_return_mat.download(return_mat);

  } else {

    // scale to analysis size if different
    if (input_frame_size.x() != config.resolution[0] ||
        input_frame_size.y() != config.resolution[1]) {
      auto &resolution = config.resolution;
      if (resolution[0] != 0 && resolution[1] != 0) {
        cv::resize(input_mat, return_mat, {resolution[0], resolution[1]}, 0, 0,
                   cv::INTER_LINEAR);
      } else {
        return_mat = input_mat;
      }
    } else {
      return_mat = input_mat;
    }

    // convert the image to grayscale
    cv::cvtColor(return_mat, return_mat, cv::COLOR_RGBA2GRAY);

    // binary colour thresholding
    cv::threshold(return_mat, return_mat, config.binary_threshold[0],
                  config.binary_threshold[1], cv::THRESH_BINARY);

    // blur the image
    if (config.blur_size > 0)
      cv::blur(return_mat, return_mat,
               cv::Size(config.blur_size, config.blur_size));

    // perform canny edge detection
    if (config.canny.enabled) {
      cv::Canny(return_mat, return_mat, config.canny.min_threshold,
                config.canny.max_threshold, config.canny.aperture_size);
    }
  }

  return std::move(return_mat);
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>,
           std::vector<uchar>>
Analyser2D::calculate_optical_flow(
    const cv::Mat &input_frame_1, const cv::Mat &input_frame_2,
    const pc::analysis::OpticalFlowConfiguration &config, bool use_cuda) {

  std::vector<cv::Point2f> feature_point_positions;
  std::vector<cv::Point2f> new_feature_point_positions;
  std::vector<uchar> status;

  if (use_cuda) {
    cv::cuda::GpuMat gpu_input_frame_1(input_frame_1);
    cv::cuda::GpuMat gpu_input_frame_2(input_frame_2);
    cv::cuda::GpuMat gpu_feature_point_positions,
        gpu_new_feature_point_positions, gpu_feature_point_status;

    // find acceptable feature points to track accross
    // the previous frame
    auto feature_point_detector = cv::cuda::createGoodFeaturesToTrackDetector(
        gpu_input_frame_2.type(), config.feature_point_count,
        config.cuda_feature_detector_quality_cutoff,
        config.feature_point_distance);
    feature_point_detector->detect(gpu_input_frame_2,
                                   gpu_feature_point_positions);

    if (!gpu_feature_point_positions.empty()) {

      // track the feature points' motion into the new frame
      auto optical_flow_filter = cv::cuda::SparsePyrLKOpticalFlow::create();
      optical_flow_filter->calc(
          gpu_input_frame_2, gpu_input_frame_1, gpu_feature_point_positions,
          gpu_new_feature_point_positions, gpu_feature_point_status);

      gpu_feature_point_positions.download(feature_point_positions);
      gpu_new_feature_point_positions.download(new_feature_point_positions);
      gpu_feature_point_status.download(status);
    }

  } else {
    // find acceptable feature points to track accross
    // the previous frame
    cv::goodFeaturesToTrack(input_frame_2, feature_point_positions,
                            config.feature_point_count, 0.01,
                            config.feature_point_distance);

    // track the feature points' motion into the new frame
    if (feature_point_positions.size() > 0) {
      cv::calcOpticalFlowPyrLK(
          input_frame_2, input_frame_1, feature_point_positions,
          new_feature_point_positions, status, cv::noArray());
    }
  }
  return {std::move(feature_point_positions),
          std::move(new_feature_point_positions), std::move(status)};
}

void Analyser2D::frame_analysis(std::stop_token stop_token) {

  while (!stop_token.stop_requested()) {

    std::optional<Magnum::Image2D> image_opt;
    std::optional<Analyser2DConfiguration> config_opt;

    {
      std::unique_lock dispatch_lock(_dispatch_mutex);

      _dispatch_condition_variable.wait(dispatch_lock, [&] {
        auto values_filled =
            (_input_image.has_value() && _input_config.has_value());
        return stop_token.stop_requested() || values_filled;
      });

      if (stop_token.stop_requested())
        break;

      // move the data onto this thread
      image_opt = std::move(_input_image);
      _input_image.reset();

      config_opt = std::move(_input_config);
      _input_config.reset();
    }

    using namespace std::chrono;

    // now we are free to process our image without holding the main thread
    auto start_time = system_clock::now();

    auto &image = *image_opt;
    auto &analysis_config = *config_opt;

    auto input_frame_size = image.size();
    if (input_frame_size.x() <= 1 || input_frame_size.y() <= 1) {
      pc::logger->warn("Analysis received invalid frame size: {}x{}",
                       input_frame_size.x(), input_frame_size.y());
      continue;
    }

    const cv::Point2i output_resolution = {input_frame_size.x(),
                                           input_frame_size.y()};
    const cv::Point2i analysis_frame_size = {analysis_config.resolution[0],
                                             analysis_config.resolution[1]};

    const auto &output_scale = analysis_config.output.scale;
    const auto &output_offset = analysis_config.output.offset;

    cv::Mat analysis_input;
    // TODO this try/catch could be replaced with better CUDA
    // synchronisation... it throws when k4a initialises it's CUDA pipeline
    // because it puts CUDA in a state where opencv cannot use it
    try {
      analysis_input = setup_input_frame(image, analysis_config);
    } catch (cv::Exception e) {
      pc::logger->error(e.what());
      continue;
    }

    // create a new RGBA image initialised to fully transparent
    cv::Mat output_mat(input_frame_size.y(), input_frame_size.x(), CV_8UC4,
                       cv::Scalar(0, 0, 0, 0));

    if (!_previous_analysis_image.has_value() ||
        _previous_analysis_image.value().size != analysis_input.size) {
      // initialise the previous frame if needed
      _previous_analysis_image = analysis_input;
    }

    std::vector<gui::OverlayText> frame_labels;

    // find the countours in the frame
    const auto &contours = analysis_config.contours;
    std::vector<std::vector<cv::Point>> contour_list;
    cv::findContours(analysis_input, contour_list, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    const auto raw_contour_count = contour_list.size();

    // TODO use serializable types all the way through instead of
    // doubling up with the opencv types and std library types

    // normalise the contours
    std::vector<std::vector<cv::Point2f>> contour_list_norm;
    contour_list_norm.reserve(raw_contour_count);
    // keep a std copy to easily serialize for publishing
    std::vector<std::vector<std::array<float, 2>>> contour_list_std;
    if (contours.publish) {
      contour_list_std.reserve(raw_contour_count);
    }

    auto calculate_centroids = contours.publish_centroids || contours.label;
    std::list<std::array<float, 2>> centroids;
    std::list<double> centroid_contour_areas;

    std::vector<cv::Point2f> norm_contour;
    for (auto &contour : contour_list) {
      norm_contour.clear();

      for (auto &point : contour) {
        norm_contour.push_back(
            {point.x / static_cast<float>(analysis_config.resolution[0]),
             point.y / static_cast<float>(analysis_config.resolution[1])});
      }

      bool use_contour = true;

      const double area = cv::contourArea(norm_contour);

      if (contours.simplify) {
        // remove contours that are too small
        if (area < contours.simplify_min_area / 1000) {
          use_contour = false;
          continue;
        }
        // simplify remaining shapes
        const double epsilon = (contours.simplify_arc_scale / 10) *
                               cv::arcLength(norm_contour, false);
        cv::approxPolyDP(norm_contour, norm_contour, epsilon, false);
      }

      if (!use_contour)
        continue;

      contour_list_norm.push_back(norm_contour);

      if (contours.publish) {
        // if we're publishing shapes we need to use std types
        std::vector<std::array<float, 2>> contour_std;
        for (auto &point : norm_contour) {
          auto output_x = point.x * output_scale[0] + output_offset[0];
          auto output_y =
              output_scale[1] - (point.y * output_scale[1] - output_offset[1]);
          contour_std.push_back({output_x, output_y});
        }
        contour_list_std.push_back(contour_std);
      }

      if (calculate_centroids) {
        // get the centroid
        auto m = cv::moments(norm_contour);
        std::array<float, 2> centroid{static_cast<float>(m.m10 / m.m00),
                                      1 - static_cast<float>(m.m01 / m.m00)};
        // and insert it into our centroids list based on descending contour
        // area size
        auto &a = centroid_contour_areas;
        auto it = std::lower_bound(a.begin(), a.end(), area, std::greater<>());
        auto index = std::distance(a.begin(), it);
        a.insert(it, area);
        // so the biggest contours are first in the centroid list
        auto list_it = centroids.begin();
        std::advance(list_it, index);
        centroids.insert(list_it, centroid);
      }
    }

    if (stop_token.stop_requested())
      break;

    std::initializer_list<std::string_view> address_nodes = {_host->name(),
                                                             "analyser_2d"};

    if (contours.publish) {
      publisher::publish_all("contours", contour_list_std, address_nodes);
    }

    if (contours.publish_centroids) {
      publisher::publish_all("centroids", centroids, address_nodes);
    }

    // Scale our contours to the output image size if we intend to draw them
    std::vector<std::vector<cv::Point>> contour_list_scaled;
    std::vector<cv::Point> scaled_contour;
    if (contours.draw) {
      contour_list_scaled.reserve(contour_list_norm.size());
      for (auto &contour : contour_list_norm) {
        scaled_contour.clear();
        scaled_contour.reserve(contour.size());
        for (auto &point : contour) {
          scaled_contour.emplace_back(
              static_cast<int>(point.x * output_resolution.x),
              static_cast<int>(point.y * output_resolution.y));
        }
        contour_list_scaled.push_back(scaled_contour);
      }
    }
    // Scale our centroids if we need them for drawing
    auto scale_centroids = contours.label;
    std::vector<cv::Point> centroids_scaled;
    if (scale_centroids) {
      centroids_scaled.reserve(centroids.size());
      std::size_t i = 0;
      for (const auto &centroid : centroids) {
        auto x = static_cast<int>(centroid[0] * output_resolution.x);
        auto y = static_cast<int>(centroid[1] * output_resolution.y);
        centroids_scaled.emplace_back(x, y);
        if (contours.label) {
          frame_labels.push_back({fmt::format("c_{}", i), {x, y}});
        }
        i++;
      }
    }

    // triangulation for building polygons from contours
    auto &triangulate = analysis_config.contours.triangulate;

    std::vector<std::array<cv::Point2f, 3>> triangles;
    std::vector<std::array<float, 2>> vertices;

    if (triangulate.enabled) {

      for (auto &contour : contour_list_norm) {

        std::vector<std::vector<std::pair<float, float>>> polygon(
            contour.size());
        for (auto &point : contour) {
          polygon[0].push_back({point.x, point.y});
        }
        auto indices = mapbox::earcut<int>(polygon);

        std::array<cv::Point2f, 3> triangle;

        auto vertex = 0;
        for (const auto &index : indices) {
          auto point =
              cv::Point2f(polygon[0][index].first, polygon[0][index].second);

          triangle[vertex % 3] = {polygon[0][index].first,
                                  polygon[0][index].second};

          if (++vertex % 3 == 0) {

            float a = cv::norm(triangle[1] - triangle[0]);
            float b = cv::norm(triangle[2] - triangle[1]);
            float c = cv::norm(triangle[2] - triangle[0]);
            float s = (a + b + c) / 2.0f;
            float area = std::sqrt(s * (s - a) * (s - b) * (s - c));

            if (area >= triangulate.minimum_area) {

              triangles.push_back(triangle);

              if (triangulate.publish) {
                for (int i = 0; i < 3; i++) {
                  auto output_x =
                      triangle[i].x * output_scale[0] + output_offset[0];
                  auto output_y =
                      output_scale[1] -
                      (triangle[i].y * output_scale[1] - output_offset[1]);
                  vertices.push_back({output_x, output_y});
                }
              }

              if (triangulate.draw) {
                static const cv::Scalar triangle_fill(120, 120, 60, 120);
                static const cv::Scalar triangle_border(255, 255, 0, 255);

                std::array<cv::Point, 3> triangle_scaled;
                std::transform(
                    triangle.begin(), triangle.end(), triangle_scaled.begin(),
                    [&](auto &point) {
                      return cv::Point{
                          static_cast<int>(point.x * output_resolution.x),
                          static_cast<int>(point.y * output_resolution.y)};
                    });

                cv::fillConvexPoly(output_mat, triangle_scaled.data(), 3,
                                   triangle_fill);
                cv::polylines(output_mat, triangle_scaled, true,
                              triangle_border, 1, cv::LINE_AA);
              }
            }

            vertex = 0;
          }
        }
      }

      if (triangulate.publish && !vertices.empty()) {
        if (vertices.size() < 3) {
          pc::logger->warn("Invalid triangle vertex count: {}",
                           vertices.size());
        } else {
	  publisher::publish_all("triangles", vertices, address_nodes);
        }
      }
    }

    if (contours.draw) {
      const cv::Scalar line_colour{255, 0, 0, 255};
      constexpr auto line_thickness = 2;
      cv::drawContours(output_mat, contour_list_scaled, -1, line_colour,
                       line_thickness, cv::LINE_AA);
    }

    // optical flow
    const auto &optical_flow = analysis_config.optical_flow;
    if (optical_flow.enabled) {

      static const cv::Scalar line_colour{200, 200, 255, 200};

      struct FlowVector {
        std::array<float, 2> position;
        std::array<float, 2> magnitude;
        std::array<float, 4> array() const {
          return {position[0], position[1], magnitude[0], magnitude[1]};
        }
      };

      auto [feature_point_positions, new_feature_point_positions,
            feature_point_status] =
          calculate_optical_flow(
              analysis_input, _previous_analysis_image.value(),
              analysis_config.optical_flow, analysis_config.use_cuda);

      std::vector<FlowVector> flow_field;
      // collect data as flattened layout as well
      // (easier to serialize for publishing)
      std::vector<std::array<float, 4>> flow_field_flattened;

      for (size_t i = 0; i < feature_point_positions.size(); ++i) {
        if (feature_point_status[i] == false)
          continue;
        cv::Point2f start = {feature_point_positions[i].x,
                             feature_point_positions[i].y};
        cv::Point2f end = {new_feature_point_positions[i].x,
                           new_feature_point_positions[i].y};

        cv::Point2f distance{end.x - start.x, end.y - start.y};
        float magnitude =
            std::sqrt(distance.x * distance.x + distance.y * distance.y);
        float scaled_magnitude =
            std::pow(magnitude, optical_flow.magnitude_exponent) *
            optical_flow.magnitude_scale;

        cv::Point2f scaled_distance{distance.x * scaled_magnitude / magnitude,
                                    distance.y * scaled_magnitude / magnitude};

        cv::Point2f normalised_position{start.x / analysis_frame_size.x,
                                        start.y / analysis_frame_size.y};

        cv::Point2f normalised_distance{
            scaled_distance.x / analysis_frame_size.x,
            scaled_distance.y / analysis_frame_size.y};

        cv::Point2f absolute_distance{std::abs(normalised_distance.x),
                                      std::abs(normalised_distance.y)};

        if (absolute_distance.x > optical_flow.minimum_distance &&
            absolute_distance.y > optical_flow.minimum_distance &&
            absolute_distance.x < optical_flow.maximum_distance &&
            absolute_distance.y < optical_flow.maximum_distance) {

          FlowVector flow_vector;
          flow_vector.position = {normalised_position.x, normalised_position.y};

          flow_vector.magnitude = {normalised_distance.x,
                                   normalised_distance.y};

          flow_field.push_back(flow_vector);

          if (optical_flow.publish) {
            auto flat_vector = flow_vector.array();
            flat_vector[0] =
                flat_vector[0] * output_scale[0] + output_offset[0];
            flat_vector[1] =
                output_scale[1] -
                (flat_vector[1] * output_scale[1] - output_offset[1]);
            flow_field_flattened.push_back(flat_vector);
          }

          if (optical_flow.draw) {
            cv::Point2f normalised_line_end = {
                normalised_position.x + normalised_distance.x,
                normalised_position.y + normalised_distance.y};
            cv::Point2f line_start{normalised_position.x * output_resolution.x,
                                   normalised_position.y * output_resolution.y};
            cv::Point2f line_end{normalised_line_end.x * output_resolution.x,
                                 normalised_line_end.y * output_resolution.y};
            cv::line(output_mat, line_start, line_end, line_colour, 3,
                     cv::LINE_AA);
          }
        }
      }

      if (optical_flow.publish) {
	publisher::publish_all("flow", flow_field_flattened, address_nodes);
      }
    }

    if (stop_token.stop_requested())
      break;

    // copy the resulting cv::Mat data into our buffer data container
    auto element_count = output_mat.total() * output_mat.channels();
    std::unique_lock lock(_analysis_frame_buffer_data_mutex);
    _analysis_frame_buffer_data =
        Corrade::Containers::Array<uint8_t>(Corrade::NoInit, element_count);

    std::copy(output_mat.datastart, output_mat.dataend,
              _analysis_frame_buffer_data.data());

    _previous_analysis_image = analysis_input;

    // copy any frame text labels
    std::lock_guard<std::mutex> labels_lock(_analysis_labels_mutex);
    _analysis_labels = std::move(frame_labels);

    auto end_time = system_clock::now();
    _analysis_time = duration_cast<milliseconds>(end_time - start_time);

    _analysis_frame_buffer_updated = true;
  }
}

Magnum::GL::Texture2D &Analyser2D::analysis_frame() {
  std::lock_guard lock_frame(_analysis_frame_mutex);
  if (_analysis_frame_buffer_updated) {
    std::lock_guard lock_buffer(_analysis_frame_buffer_data_mutex);
    // move updated buffer data into our analysis frame Texture2D...
    // we first need to create an OpenGL Buffer for it
    if (_analysis_frame_buffer_data.size() !=
        _frame_size.x() * _frame_size.y() * 4) {
      pc::logger->warn("Analysis framebuffer size mismatch");
      return *_analysis_frame;
    }
    Magnum::GL::BufferImage2D buffer{Magnum::GL::PixelFormat::RGBA,
                                     Magnum::GL::PixelType::UnsignedByte,
                                     _frame_size, _analysis_frame_buffer_data,
                                     Magnum::GL::BufferUsage::StaticDraw};
    // then we can set the data
    _analysis_frame->setSubImage(0, {}, buffer);
  }
  return *_analysis_frame;
}

int Analyser2D::analysis_time() {
  std::chrono::milliseconds time = _analysis_time;
  if (time.count() > 0)
    return time.count();
  else
    return 1;
}

std::vector<gui::OverlayText> Analyser2D::analysis_labels() {
  std::lock_guard<std::mutex> lock(_analysis_labels_mutex);
  return _analysis_labels;
}

} // namespace pc::analysis
