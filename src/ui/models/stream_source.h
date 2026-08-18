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

  Q_PROPERTY(bool isRadius READ isRadius NOTIFY contentChanged)
  Q_PROPERTY(qreal radiusMetres READ radiusMetres NOTIFY contentChanged)

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
    return switched_on() ? read_value<pc::position_bounds>() : std::nullopt;
  }
  std::optional<pc::radius> radius() override {
    return switched_on() ? read_value<pc::radius>() : std::nullopt;
  }

  // a value carrying a switch registers it alongside itself
  bool switched_on() const {
    const auto active = _registry.get(_path + "/active");
    if (!active) return true;
    return pc::from_config_value<bool>(*active);
  }

  bool isRadius() const { return read_value<pc::radius>().has_value(); }

  qreal radiusMetres() const {
    const auto held = switched_on() ? read_value<pc::radius>() : std::nullopt;
    return held ? held->metres() : 0.0;
  }

  void notifyChanged() { emit contentChanged(); }

signals:
  void contentChanged();

private:
  pc::ConfigRegistry &_registry;
  std::string _path;

  template <class T> std::optional<T> read_value() const {
    const auto value = _registry.get(_path);
    if (!value) return std::nullopt;
    const auto *held = std::get_if<T>(&value.value());
    if (!held) return std::nullopt;
    return *held;
  }

  template <class T> T read() {
    const auto value = _registry.get(_path);
    if (!value) return nullptr;
    const auto *stream = std::get_if<T>(&value.value());
    return stream ? *stream : nullptr;
  }
};
