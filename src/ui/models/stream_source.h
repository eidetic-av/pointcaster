#pragma once

#include "stream_adapter.h"

#include <QObject>
#include <QString>
#include <config/config_registry.h>
#include <optional>
#include <string>
#include <variant>

// reads whatever was last written into a config registry path 
class StreamSource : public QObject, public StreamAdapter {
  Q_OBJECT

  Q_PROPERTY(QString path READ path CONSTANT)

public:
  StreamSource(pc::ConfigRegistry &registry, std::string path,
               QObject *parent = nullptr)
      : QObject(parent), _registry(registry), _path(std::move(path)) {}

  Q_INVOKABLE StreamAdapter *streamAdapter() {
    return static_cast<StreamAdapter *>(this);
  }

  QString path() const { return QString::fromStdString(_path); }

  pc::AabbListPtr aabb_list() override { return read<pc::AabbListPtr>(); }
  pc::VoxelisedCloudPtr voxelised_cloud() override {
    return read<pc::VoxelisedCloudPtr>();
  }
  std::optional<pc::position_bounds> bounds() override {
    const auto value = _registry.get(_path);
    if (!value) return std::nullopt;
    const auto *bounds = std::get_if<pc::position_bounds>(&value.value());
    if (!bounds) return std::nullopt;
    return *bounds;
  }

  void notifyChanged() { emit contentChanged(); }

signals:
  void contentChanged();

private:
  pc::ConfigRegistry &_registry;
  std::string _path;

  template <class T> T read() {
    const auto value = _registry.get(_path);
    if (!value) return nullptr;
    const auto *stream = std::get_if<T>(&value.value());
    return stream ? *stream : nullptr;
  }
};
