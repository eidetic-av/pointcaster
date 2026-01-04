#pragma once

#include <QObject>
#include <QString>
#include <QVariant>

class ConfigAdapter : public QObject {
  Q_OBJECT
public:
  explicit ConfigAdapter(QObject *parent = nullptr) : QObject(parent) {}

  ~ConfigAdapter() override = default;

  // metadata required for each adapter to generate about its configs members

  Q_INVOKABLE virtual QString configType() const = 0;

  Q_INVOKABLE virtual int fieldCount() const = 0;

  Q_INVOKABLE virtual QString fieldName(int index) const = 0;

  Q_INVOKABLE virtual QString fieldTypeName(int index) const = 0;

  Q_INVOKABLE virtual QVariant fieldValue(int index) const = 0;

  Q_INVOKABLE virtual void setFieldValue(int index, const QVariant &value) = 0;

  Q_INVOKABLE virtual QString displayName() const { return {}; }

  // TODO is this needed?
  // default expression for the field at index as a string, e.g. "{0.5f}"
  Q_INVOKABLE virtual QString defaultExpression(int index) const {
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

signals:
  void fieldChanged(int index);
};