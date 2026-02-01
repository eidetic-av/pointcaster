#pragma once

#include <config/config_variant.h>

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariant>

class ConfigAdapter : public QObject {
  Q_OBJECT

public:
  explicit ConfigAdapter(QObject *parent = nullptr) : QObject(parent) {}
  ~ConfigAdapter() override = default;

  Q_INVOKABLE virtual QString configType() const = 0;
  Q_INVOKABLE virtual QString displayName() const { return {}; }

  // stable paths to each member like:
  //  - "id"
  //  - "camera/locked"
  Q_INVOKABLE virtual QStringList fieldPaths() const = 0;

  // ---- Path-based value access
  // Path segments use "/" as separator, e.g. "camera/locked".
  Q_INVOKABLE virtual QVariant value(const QString &path) const = 0;

  // UI uses this... WorkspaceModel listens to editRequested() and pushes undo
  // commands
  Q_INVOKABLE void set(const QString &path, const QVariant &value) {
    emit editRequested(path, value);
  }

  // Mutating apply that directly changes the referenced config.
  // Returns true if a change actually happened.
  Q_INVOKABLE virtual bool apply(const QString &path,
                                 const QVariant &value) = 0;

  // ---- defautl metadata is empty
  Q_INVOKABLE virtual QString typeName(const QString &path) const {
    Q_UNUSED(path);
    return {};
  }

  Q_INVOKABLE virtual QVariant defaultValue(const QString &path) const {
    Q_UNUSED(path);
    return {};
  }

  Q_INVOKABLE virtual QVariant minMax(const QString &path) const {
    Q_UNUSED(path);
    return {};
  }

  Q_INVOKABLE virtual bool isOptional(const QString &path) const {
    Q_UNUSED(path);
    return false;
  }

  Q_INVOKABLE virtual bool isDisabled(const QString &path) const {
    Q_UNUSED(path);
    return false;
  }

  Q_INVOKABLE virtual bool isHidden(const QString &path) const {
    Q_UNUSED(path);
    return false;
  }

  Q_INVOKABLE virtual bool isEnum(const QString &path) const {
    Q_UNUSED(path);
    return false;
  }

  Q_INVOKABLE virtual QVariant enumOptions(const QString &path) const {
    Q_UNUSED(path);
    return {};
  }

  void notifyFieldChanged(const QString &path) { emit fieldChanged(path); }

  virtual bool setConfig(const pc::ConfigurationVariant &) = 0;

signals:
  void editRequested(const QString &path, const QVariant &value);
  void fieldChanged(const QString &path);
};