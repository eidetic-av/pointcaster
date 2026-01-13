#pragma once

#include "../../plugins/devices/device_plugin.h"
#include "device_status.h"
#include <QObject>
#include <QString>
#include <QVariant>

// TODO maybe this should be a DeviceConfigAdapter...

class ConfigAdapter : public QObject {
  Q_OBJECT
  Q_PROPERTY(
      pc::ui::WorkspaceDeviceStatus status READ status NOTIFY statusChanged)
public:
  pc::ui::WorkspaceDeviceStatus status() const { return _status; }

  explicit ConfigAdapter(pc::devices::DevicePlugin *plugin,
                         QObject *parent = nullptr)
      : QObject(parent), _plugin(plugin) {}

  ~ConfigAdapter() override = default;

  // metadata required for each adapter to generate about its configs members

  Q_INVOKABLE virtual QString configType() const = 0;

  Q_INVOKABLE virtual int fieldCount() const = 0;

  Q_INVOKABLE virtual QString fieldName(int index) const = 0;

  Q_INVOKABLE virtual QString fieldTypeName(int index) const = 0;

  Q_INVOKABLE virtual QVariant fieldValue(int index) const = 0;

  Q_INVOKABLE virtual void setFieldValue(int index, const QVariant &value) = 0;

  Q_INVOKABLE virtual QString displayName() const { return {}; }

  Q_INVOKABLE virtual QVariant defaultValue(int index) const {
    Q_UNUSED(index);
    return {};
  }

  // min/max range for the field at index.
  Q_INVOKABLE virtual QVariant minMax(int index) const {
    Q_UNUSED(index);
    return {};
  }

  Q_INVOKABLE virtual bool isOptional(int index) const {
    Q_UNUSED(index);
    return false;
  }

  Q_INVOKABLE virtual bool isDisabled(int index) const {
    Q_UNUSED(index);
    return false;
  }

  Q_INVOKABLE virtual bool isHidden(int index) const {
    Q_UNUSED(index);
    return false;
  }

  void setStatusFromCore(pc::devices::DeviceStatus s) {
    const auto q = pc::ui::toQt(s);
    if (q == _status) return;
    _status = q;
    emit statusChanged();
  }

signals:
  void fieldChanged(int index);
  void statusChanged();

protected:
  pc::devices::DevicePlugin *_plugin = nullptr;
  pc::ui::WorkspaceDeviceStatus _status =
      pc::ui::WorkspaceDeviceStatus::Unloaded;
};