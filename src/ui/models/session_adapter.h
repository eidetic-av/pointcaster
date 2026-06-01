#pragma once

#include "camera_image_provider.h"
#include "point_cloud_adapter.h"

#include <QImage>
#include <QObject>
#include <QStringList>
#include <QVariantMap>
#include <memory>
#include <session/session.h>

class SessionAdapter : public QObject, public PointCloudAdapter {
  Q_OBJECT

  Q_PROPERTY(QStringList frameSlots READ frameSlots NOTIFY frameSlotsChanged)
  Q_PROPERTY(QVariantMap frameUrls READ frameUrls NOTIFY frameUrlsChanged)

public:
  explicit SessionAdapter(pc::Session *session,
                          CameraImageProvider *imageProvider,
                          QObject *parent = nullptr)
      : QObject(parent), _session(session), _imageProvider(imageProvider) {}

  ~SessionAdapter() override {
    if (_imageProvider) _imageProvider->removeFrames(keyPrefix());
  }

  // ---- PointCloudAdapter ----

  Q_INVOKABLE std::shared_ptr<pc::PointCloud> point_cloud() override {
    return _session ? _session->point_cloud() : nullptr;
  }

  Q_INVOKABLE std::shared_ptr<std::vector<std::byte>> render_data() override {
    return _session ? _session->render_data() : nullptr;
  }

  // Return `this` as PointCloudAdapter* so QML can pass it to PointCloudGeometry.
  Q_INVOKABLE PointCloudAdapter *pointCloudAdapter() {
    return static_cast<PointCloudAdapter *>(this);
  }

  // ---- Frame slots / URLs (mirrors DeviceAdapter::syncCameraFrames) ----

  QStringList frameSlots() const { return _frameSlotNames; }
  QVariantMap frameUrls() const { return _frameUrls; }

  // ---- Called on Qt UI thread via QMetaObject::invokeMethod from the callback ----

  void notifyPointCloudUpdated() {
    emit pointCloudUpdated();
    syncCameraFrames();
  }

  void syncCameraFrames() {
    if (!_session || !_imageProvider) return;

    auto frames = _session->latest_camera_frames();

    if (frames.empty()) {
      if (!_frameSlotNames.isEmpty()) {
        _imageProvider->removeFrames(keyPrefix());
        _frameSlotNames.clear();
        _frameUrls.clear();
        emit frameSlotsChanged();
        emit frameUrlsChanged();
      }
      return;
    }

    const auto prefix = keyPrefix();
    QStringList newSlotNames;
    QVariantMap newUrls;

    for (const auto &frame : frames) {
      if (!frame.frame_data) continue;
      auto &data = *frame.frame_data.value();

      const auto w = static_cast<int>(data.width);
      const auto h = static_cast<int>(data.height);
      const auto &colors = data.main_color_buffer();

      QImage img(reinterpret_cast<const uchar *>(colors.data()), w, h, w * 4,
                 QImage::Format_RGBA8888);
      img = img.copy();

      const auto name = QString::fromStdString(frame.name);
      const auto key = prefix + name;

      // Overwrite atomically; no removeFrames gap (same reasoning as DeviceAdapter).
      _imageProvider->registerFrame(key, img);
      newSlotNames.append(name);
      newUrls[name] = QStringLiteral("image://camera/") + key + "?" +
                      QString::number(_frameRevision);
    }

    _frameRevision++;

    if (newSlotNames != _frameSlotNames) {
      _frameSlotNames = std::move(newSlotNames);
      emit frameSlotsChanged();
    }
    _frameUrls = std::move(newUrls);
    emit frameUrlsChanged();
  }

  pc::Session *session() const { return _session; }

signals:
  void pointCloudUpdated();
  void frameSlotsChanged();
  void frameUrlsChanged();

private:
  pc::Session *_session = nullptr;
  CameraImageProvider *_imageProvider = nullptr;

  QStringList _frameSlotNames;
  QVariantMap _frameUrls;
  int _frameRevision = 0;

  QString keyPrefix() const {
    return _session ? QString::fromStdString(_session->id) + "/" : QString{};
  }
};