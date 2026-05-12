#pragma once

#include <QImage>
#include <QQuickImageProvider>
#include <functional>
#include <mutex>
#include <unordered_map>

// TODO need to look over this api...
// is it necessary? can my operator adapter itself also extend from QQuickImageProvider?

// stores images keyed by "deviceIndex/operatorIndex/projectionName"
class ProjectionImageProvider : public QQuickImageProvider {
public:
  ProjectionImageProvider() : QQuickImageProvider(QQuickImageProvider::Image) {
    _instance = this;
  }

  static ProjectionImageProvider *instance() { return _instance; }

  void update(const QString &key, const QImage &img) {
    std::lock_guard lock(_mutex);
    _images[key] = img;
  }

  void remove(const QString &keyPrefix) {
    std::lock_guard lock(_mutex);
    for (auto it = _images.begin(); it != _images.end();) {
      if (it->first.startsWith(keyPrefix))
        it = _images.erase(it);
      else
        ++it;
    }
  }

  QImage requestImage(const QString &id, QSize *size,
                      const QSize & /*requestedSize*/) override {
    const auto clean = id.section('?', 0, 0);
    std::lock_guard lock(_mutex);
    auto it = _images.find(clean);
    if (it == _images.end()) {
      if (size) *size = QSize(0, 0);
      return {};
    }
    if (size) *size = it->second.size();
    return it->second;
  }

private:
  static inline ProjectionImageProvider *_instance = nullptr;
  std::mutex _mutex;
  std::unordered_map<QString, QImage> _images;
};