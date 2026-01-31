#pragma once

#include <config_variant.h>

#include <QObject>
#include <QString>
#include <QVariant>

class ConfigAdapter : public QObject {
  Q_OBJECT

public:
  explicit ConfigAdapter(QObject *parent = nullptr) : QObject(parent) {}
  ~ConfigAdapter() override = default;

  Q_INVOKABLE virtual QString configType() const = 0;
  Q_INVOKABLE virtual int fieldCount() const = 0;
  Q_INVOKABLE virtual QString fieldName(int index) const = 0;
  Q_INVOKABLE virtual QString fieldTypeName(int index) const = 0;

  Q_INVOKABLE virtual QVariant fieldValue(int index) const = 0;
  Q_INVOKABLE virtual void setFieldValue(int index, const QVariant &value) = 0;

  Q_INVOKABLE virtual bool applyFieldValue(int index,
                                           const QVariant &value) = 0;

  Q_INVOKABLE virtual QString displayName() const { return {}; }

  Q_INVOKABLE virtual QVariant defaultValue(int index) const { return {}; }
  Q_INVOKABLE virtual QVariant minMax(int index) const { return {}; }
  Q_INVOKABLE virtual bool isOptional(int index) const { return false; }
  Q_INVOKABLE virtual bool isDisabled(int index) const { return false; }
  Q_INVOKABLE virtual bool isHidden(int index) const { return false; }
  Q_INVOKABLE virtual bool isEnum(int index) const { return false; }
  Q_INVOKABLE virtual QVariant enumOptions(int index) const { return {}; }

  void notifyFieldChanged(int index) { emit fieldChanged(index); }

  // Core-only
  virtual bool setConfig(const pc::CoreConfigurationVariant &) = 0;

signals:
  void fieldEditRequested(int index, const QVariant &value);
  void fieldChanged(int index);
};
