#pragma once

#include <QImage>
#include <QQuickImageProvider>
#include <mutex>
#include <unordered_map>

// Registered once with the QML engine. Populated by DeviceAdapter
// from pipeline camera frames.
class CameraImageProvider : public QQuickImageProvider {
public:
  CameraImageProvider() : QQuickImageProvider(QQuickImageProvider::Image) {}

  void registerFrame(const QString &key, const QImage &img) {
    std::lock_guard lock(_mutex);
    _frames[key] = img;
  }

  void removeFrames(const QString &keyPrefix) {
    std::lock_guard lock(_mutex);
    for (auto it = _frames.begin(); it != _frames.end();) {
      if (it->first.startsWith(keyPrefix))
        it = _frames.erase(it);
      else
        ++it;
    }
  }

  QImage requestImage(const QString &id, QSize *size,
                      const QSize & /*requestedSize*/) override {
    const auto clean = id.section('?', 0, 0);
    std::lock_guard lock(_mutex);
    auto it = _frames.find(clean);
    if (it == _frames.end()) {
      if (size) *size = QSize(0, 0);
      return {};
    }
    if (size) *size = it->second.size();
    return it->second;
  }

private:
  std::mutex _mutex;
  std::unordered_map<QString, QImage> _frames;
};