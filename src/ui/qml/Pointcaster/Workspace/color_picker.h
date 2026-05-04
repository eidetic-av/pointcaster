#pragma once

#include <QObject>
#include <QtQmlIntegration/qqmlintegration.h>
#include <logger/logger.h>

namespace pc::ui::qml {

class ColorPicker : public QObject {
  Q_OBJECT
  QML_NAMED_ELEMENT(ColorPicker)

public:
  explicit ColorPicker(QObject *parent = nullptr) : QObject(parent) {}

  Q_INVOKABLE void run(int x, int y) {
    pc::logger()->debug("{}, {}", x, y);
  }
};

} // namespace pc::ui::qml