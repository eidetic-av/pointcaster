#pragma once
#include "projection_image_provider.h"
#include <QObject>
#include <QStringList>
#include <camera/camera_frame.h>
#include <plugins/operators/operator_plugin.h>
#include <plugins/operators/operator_variants.h>
#include <ranges>

class OperatorAdapter : public QObject {
  Q_OBJECT
  Q_PROPERTY(QStringList frameSources READ frameSources NOTIFY framesUpdated)

public:
  OperatorAdapter(pc::operators::OperatorPlugin *plugin,
                  QObject *parent = nullptr)
      : QObject(parent), _plugin(plugin) {}

  ~OperatorAdapter() {
    if (auto *provider = ProjectionImageProvider::instance()) {
      provider->remove(keyPrefix());
    }
  }

  QStringList frameSources() const { return _sources; }

  void syncCameraFrames() {
    // TODO this provider stuff... i think the adapter itself can be a provider
    auto *provider = ProjectionImageProvider::instance();
    if (!provider) return;

    auto camera_frames = _plugin->camera_frames();
    _sources.clear();

    const auto prefix = keyPrefix();

    auto valid_camera_frames =
        camera_frames |
        std::views::transform(
            [](auto ref) -> pc::camera::CameraFrame & { return ref.get(); }) |
        std::views::filter(
            [](const auto &frame) { return frame.frame_data.has_value(); });

    for (auto &frame : valid_camera_frames) {
      auto &f = *frame.frame_data;
      // TODO here we should expose sources for all available frames not just
      // color buffer

      auto &color_buffer = f.color_buffers["color"];

      QImage img(f.width, f.height, QImage::Format_RGBA8888);
      std::memcpy(img.bits(), color_buffer.data(), f.width * f.height * 4);

      auto key = QStringLiteral("%1%2").arg(
          prefix, QString::fromUtf8(frame.name.data(), frame.name.size()));

      provider->update(key, img);
      _sources.append(
          QStringLiteral("image://frame/%1?rev=%2").arg(key).arg(_revision));
    }

    _revision++;
    emit framesUpdated();
  }

  pc::operators::OperatorPlugin *plugin() const { return _plugin; }

signals:
  void framesUpdated();

private:
  QString keyPrefix() const {
    const auto &v = _plugin->config();
    // TODO??
    auto [id, _] = pc::operators::operator_info_from_variant(v);
    return QString::fromStdString(std::string(id)) + "/";
  }

  pc::operators::OperatorPlugin *_plugin = nullptr;
  int _revision = 0;
  QStringList _sources;
};