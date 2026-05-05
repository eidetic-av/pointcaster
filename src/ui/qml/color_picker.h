#pragma once

#include <QColor>
#include <QObject>
#include <QQuickItem>
#include <QQuickItemGrabResult>
#include <QtQmlIntegration/qqmlintegration.h>
#include <logger/logger.h>

namespace pc::ui::qml {
Q_NAMESPACE

class ColorPicker : public QObject {
  Q_OBJECT
  QML_NAMED_ELEMENT(ColorPicker)

  Q_PROPERTY(int r READ r NOTIFY picked)
  Q_PROPERTY(int g READ g NOTIFY picked)
  Q_PROPERTY(int b READ b NOTIFY picked)
  Q_PROPERTY(int a READ a NOTIFY picked)

public:
  explicit ColorPicker(QObject *parent = nullptr) : QObject(parent) {}

  Q_INVOKABLE void pick(QQuickItem *target, int x, int y);

  int r() const { return _r; };
  int g() const { return _g; };
  int b() const { return _b; };
  int a() const { return _a; };

signals:
  void picked();

private:
  int _r;
  int _g;
  int _b;
  int _a;
};

} // namespace pc::ui::qml