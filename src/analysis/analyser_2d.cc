#include "analyser_2d.h"
#include "../logger.h"
#include <Corrade/Containers/Array.h>
#include <Magnum/GL/BufferImage.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/PixelFormat.h>
#include <mapbox/earcut.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#if WITH_MQTT
#include "../mqtt/mqtt_client.h"
#endif

namespace pc::analysis {

Analyser2D::Analyser2D()
    : _analysis_thread(&Analyser2D::frame_analysis, this) {}

void Analyser2D::set_frame_size(Magnum::Vector2i frame_size) {
  std::lock_guard lock(_analysis_frame_mutex);
  _analysis_frame = std::make_unique<Magnum::GL::Texture2D>();
  _analysis_frame->setStorage(1, Magnum::GL::TextureFormat::RGBA8, frame_size);
  _frame_size = frame_size;
}

void Analyser2D::dispatch_analysis(Magnum::GL::Texture2D &texture,
				   Analyser2DConfiguration &config) {
  if (!config.enabled) return;
  std::lock_guard lock_dispatch(_dispatch_mutex);
  // move the image onto the CPU for use in our analysis thread
  _input_image = texture.image(0, Magnum::Image2D{Magnum::PixelFormat::RGBA8Unorm});
  _input_config = config;
  _dispatch_condition_variable.notify_one();
}

void Analyser2D::frame_analysis(std::stop_token stop_token) {
  while (!stop_token.stop_requested()) {

    std::optional<Magnum::Image2D> image_opt;
    std::optional<pc::analysis::Analyser2DConfiguration> config_opt;

    {
      std::unique_lock dispatch_lock(_dispatch_mutex);

      _dispatch_condition_variable.wait(dispatch_lock, [&] {
        auto values_filled =
            (_input_image.has_value() && _input_config.has_value());
        return stop_token.stop_requested() || values_filled;
      });

      if (stop_token.stop_requested())
        break;

      // start move the data onto this thread
      image_opt = std::move(_input_image);
      _input_image.reset();

      config_opt = std::move(_input_config);
      _input_config.reset();
    }

    // now we are free to process our image without holding the main thread
    auto start_time = std::chrono::system_clock::now();

    auto &image = *image_opt;
    auto input_frame_size = image.size();
    auto &analysis_config = *config_opt;

    if (input_frame_size.x() <= 1 || input_frame_size.y() <= 1) {
      pc::logger->warn("Analysis received invalid frame size: {}x{}",
                       input_frame_size.x(), input_frame_size.y());
      continue;
    }

    // create a new RGBA image initialised to fully transparent
    cv::Mat output_mat(input_frame_size.y(), input_frame_size.x(), CV_8UC4,
                       cv::Scalar(0, 0, 0, 0));

    // start by wrapping the input image data in an opencv matrix type
    cv::Mat input_mat(input_frame_size.y(), input_frame_size.x(), CV_8UC4,
                      image.data());

    // and scale to analysis size if different
    cv::Mat scaled_mat;
    if (input_frame_size.x() != analysis_config.resolution[0] ||
        input_frame_size.y() != analysis_config.resolution[1]) {
      auto &resolution = analysis_config.resolution;
      if (resolution[0] != 0 && resolution[1] != 0) {
        cv::resize(input_mat, scaled_mat, {resolution[0], resolution[1]}, 0, 0,
                   cv::INTER_LINEAR);
      } else {
        scaled_mat = input_mat;
      }
    } else {
      scaled_mat = input_mat;
    }

    const cv::Point2f output_scale = {
        static_cast<float>(input_mat.cols) / scaled_mat.cols,
        static_cast<float>(input_mat.rows) / scaled_mat.rows};
    const cv::Point2i output_resolution = {input_frame_size.x(),
                                           input_frame_size.y()};

    cv::Mat analysis_input(scaled_mat);
    auto analysis_frame_size = analysis_input.size;

    // convert the image to grayscale
    cv::cvtColor(analysis_input, analysis_input, cv::COLOR_RGBA2GRAY);

    // binary colour thresholding
    cv::threshold(analysis_input, analysis_input,
                  analysis_config.binary_threshold[0],
                  analysis_config.binary_threshold[1], cv::THRESH_BINARY);

    // blur the image
    if (analysis_config.blur_size > 0)
      cv::blur(analysis_input, analysis_input,
               cv::Size(analysis_config.blur_size, analysis_config.blur_size));

    // done manipulating our analysis input frame here

    if (!_previous_analysis_image.has_value() ||
        _previous_analysis_image.value().size != analysis_input.size) {
      // initialise the previous frame if needed
      _previous_analysis_image = analysis_input;
    }

    // onto analysis functions

    // // perform canny edge detection
    auto &canny = analysis_config.canny;
    if (canny.enabled) {
      cv::Canny(analysis_input, analysis_input, canny.min_threshold,
                canny.max_threshold, canny.aperture_size);
    }

    // find the countours in the frame
    auto &contours = analysis_config.contours;
    std::vector<std::vector<cv::Point>> contour_list;
    cv::findContours(analysis_input, contour_list, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // TODO use serializable types all the way through instead of
    // doubling up with the opencv types and std library types

    // normalise the contours
    std::vector<std::vector<cv::Point2f>> contour_list_norm;
    // keep a std copy to easily serialize for publishing
    std::vector<std::vector<std::array<float, 2>>> contour_list_std;

    std::vector<cv::Point2f> norm_contour;
    for (auto &contour : contour_list) {
      norm_contour.clear();

      for (auto &point : contour) {
        norm_contour.push_back(
            {point.x / static_cast<float>(analysis_config.resolution[0]),
             point.y / static_cast<float>(analysis_config.resolution[1])});
      }

      bool use_contour = true;

      if (contours.simplify) {
        // remove contours that are too small
        const double area = cv::contourArea(norm_contour);
        if (area < contours.simplify_min_area / 1000) {
          use_contour = false;
          continue;
        }
        // simplify remaining shapes
        const double epsilon = (contours.simplify_arc_scale / 10) *
                               cv::arcLength(norm_contour, false);
        cv::approxPolyDP(norm_contour, norm_contour, epsilon, false);
      }

      if (use_contour) {
        contour_list_norm.push_back(norm_contour);

        if (contours.publish) {
          // if we're publishing shapes we need to use std types
          std::vector<std::array<float, 2>> contour_std;
          for (auto &point : norm_contour) {
            contour_std.push_back({point.x, 1 - point.y});
          }
          contour_list_std.push_back(contour_std);
        }
      }
    }

    if (contours.publish) {
#if WITH_MQTT
      if (!stop_token.stop_requested())
        MqttClient::instance()->publish("contours", contour_list_std);
#endif
    }

    // create a version of the contours list scaled to the size
    // of our output image
    std::vector<std::vector<cv::Point>> contour_list_scaled;
    std::vector<cv::Point> scaled_contour;
    for (auto &contour : contour_list_norm) {
      scaled_contour.clear();
      for (auto &point : contour) {
        scaled_contour.push_back(
            {static_cast<int>(point.x * input_frame_size.x()),
             static_cast<int>(point.y * input_frame_size.y())});
      }
      contour_list_scaled.push_back(scaled_contour);
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
              vertices.push_back({triangle[0].x, 1 - triangle[0].y});
              vertices.push_back({triangle[1].x, 1 - triangle[1].y});
              vertices.push_back({triangle[2].x, 1 - triangle[2].y});

              if (triangulate.draw) {
                static const cv::Scalar triangle_fill(120, 120, 60, 120);
                static const cv::Scalar triangle_border(255, 255, 0, 255);

                std::array<cv::Point, 3> triangle_scaled;
                std::transform(
                    triangle.begin(), triangle.end(), triangle_scaled.begin(),
                    [&](auto &point) {
                      return cv::Point{
                          static_cast<int>(point.x * input_frame_size.x()),
                          static_cast<int>(point.y * input_frame_size.y())};
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
#if WITH_MQTT
          if (!stop_token.stop_requested())
            MqttClient::instance()->publish("triangles", vertices);
#endif
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
        std::array<float, 4> flipped_array() const {
          return {position[0], 1 - position[1], magnitude[0], magnitude[1]};
        }
      };

      std::vector<FlowVector> flow_field;
      // collect data as flattened layout as well
      // (easier to serialize for publishing)
      std::vector<std::array<float, 4>> flow_field_flattened;

      // find acceptable feature points to track accross
      // the previous frame
      std::vector<cv::Point2f> feature_points;
      cv::goodFeaturesToTrack(_previous_analysis_image.value(), feature_points,
                              optical_flow.feature_point_count, 0.01,
                              optical_flow.feature_point_distance);

      // track the feature points' motion into the new frame
      std::vector<cv::Point2f> new_feature_points;
      std::vector<uchar> status;
      if (feature_points.size() > 0) {
        cv::calcOpticalFlowPyrLK(_previous_analysis_image.value(),
                                 analysis_input, feature_points,
                                 new_feature_points, status, cv::noArray());
      }

      for (size_t i = 0; i < feature_points.size(); ++i) {
        if (status[i] == false)
          continue;
        cv::Point2f start = {feature_points[i].x, feature_points[i].y};
        cv::Point2f end = {new_feature_points[i].x, new_feature_points[i].y};

        cv::Point2f distance{end.x - start.x, end.y - start.y};
        float magnitude =
            std::sqrt(distance.x * distance.x + distance.y * distance.y);
        float scaled_magnitude =
            std::pow(magnitude, optical_flow.magnitude_exponent) *
            optical_flow.magnitude_scale;

        cv::Point2f scaled_distance{distance.x * scaled_magnitude / magnitude,
                                    distance.y * scaled_magnitude / magnitude};

        cv::Point2f normalised_position{start.x / analysis_frame_size[0],
                                        start.y / analysis_frame_size[1]};
        cv::Point2f normalised_distance{
            scaled_distance.x / analysis_frame_size[0],
            scaled_distance.y / analysis_frame_size[1]};

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
          flow_field_flattened.push_back(flow_vector.flipped_array());

          if (optical_flow.draw) {
            cv::Point2f scaled_end = {
                normalised_position.x + normalised_distance.x,
                normalised_position.y + normalised_distance.y};
            cv::Point2f frame_start{normalised_position.x * output_resolution.x,
                                    normalised_position.y *
                                        output_resolution.y};
            cv::Point2f frame_end{scaled_end.x * output_resolution.x,
                                  scaled_end.y * output_resolution.y};
            cv::line(output_mat, frame_start, frame_end, line_colour, 3,
                     cv::LINE_AA);
          }
        }
      }

      if (flow_field.size() > 0 && optical_flow.publish) {
#if WITH_MQTT
        if (!stop_token.stop_requested())
          MqttClient::instance()->publish("flow", flow_field_flattened);
#endif
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

    _analysis_frame_buffer_updated = true;
    _previous_analysis_image = analysis_input;

    using namespace std::chrono;
    auto end_time = system_clock::now();
    _analysis_time = duration_cast<milliseconds>(end_time - start_time);
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
    Magnum::GL::BufferImage2D buffer{
	Magnum::GL::PixelFormat::RGBA, Magnum::GL::PixelType::UnsignedByte,
	_frame_size, _analysis_frame_buffer_data, Magnum::GL::BufferUsage::StaticDraw};
    // then we can set the data
    _analysis_frame->setSubImage(0, {}, buffer);
  }
  return *_analysis_frame;
}

int Analyser2D::analysis_time() {
  std::chrono::milliseconds time = _analysis_time;
  if (time.count() > 0) return time.count();
  else return 1;
}

} // namespace pc::analysis
