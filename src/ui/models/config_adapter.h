#pragma once

#include "../../plugins/devices/device_plugin.h"
#include "device_status.h"
#include <QObject>
#include <QString>
#include <QVariant>

class ConfigAdapter : public QObject {
  Q_OBJECT
  Q_PROPERTY(
      pc::ui::WorkspaceDeviceStatus status READ status NOTIFY statusChanged)

public:
  explicit ConfigAdapter(pc::devices::DevicePlugin *plugin,
                         QObject *parent = nullptr)
      : QObject(parent), _plugin(plugin) {}

  ~ConfigAdapter() override = default;

  int deviceIndex() const { return _deviceIndex; }
  void setDeviceIndex(int index) { _deviceIndex = index; }

  pc::ui::WorkspaceDeviceStatus status() const { return _status; }

  pc::devices::DevicePlugin *plugin() const { return _plugin; }

  Q_INVOKABLE virtual QString configType() const = 0;
  Q_INVOKABLE virtual int fieldCount() const = 0;
  Q_INVOKABLE virtual QString fieldName(int index) const = 0;
  Q_INVOKABLE virtual QString fieldTypeName(int index) const = 0;

  Q_INVOKABLE virtual QVariant fieldValue(int index) const = 0;
  Q_INVOKABLE virtual void setFieldValue(int index, const QVariant &value) = 0;

  Q_INVOKABLE virtual bool applyFieldValue(int index,
                                           const QVariant &value) = 0;

  Q_INVOKABLE virtual QString displayName() const { return {}; }

  Q_INVOKABLE virtual QVariant defaultValue(int index) const {
    Q_UNUSED(index);
    return {};
  }
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
  Q_INVOKABLE virtual bool isEnum(int index) const {
    Q_UNUSED(index);
    return false;
  }
  Q_INVOKABLE virtual QVariant enumOptions(int index) const {
    Q_UNUSED(index);
    return {};
  }

  void notifyFieldChanged(int index) { emit fieldChanged(index); }

  virtual void
  setConfigFromCore(const pc::devices::DeviceConfigurationVariant &v) = 0;

  void setStatusFromCore(pc::devices::DeviceStatus s) {
    const auto q = pc::ui::toQt(s);
    if (q == _status) return;
    _status = q;
    emit statusChanged();
  }

signals:
  void fieldEditRequested(int index, const QVariant &value);
  void fieldChanged(int index);
  void statusChanged();

protected:
  pc::devices::DevicePlugin *_plugin = nullptr;
  pc::ui::WorkspaceDeviceStatus _status =
      pc::ui::WorkspaceDeviceStatus::Unloaded;
  int _deviceIndex = -1;
};
