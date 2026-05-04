#pragma once

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

public:
  explicit ColorPicker(QObject *parent = nullptr) : QObject(parent) {}

  Q_INVOKABLE void pick(QQuickItem* target, int x, int y) {
    if (target) {
      pc::logger()->trace("ColorPicker.pick(x: {}, y: {}", x, y);
      const auto result = target->grabToImage();
      QObject::connect(result.data(), &QQuickItemGrabResult::ready, [result, x, y]{
        pc::logger()->trace("grabToImage completed");
        auto image = result->image();
        auto color = image.pixelColor(x, y);
        int r, g, b, a;
        color.getRgb(&r, &g, &b, &a);
        pc::logger()->trace("r: {}, g: {}, b: {}, a: {}", color.redF(), color.greenF(), color.blueF(), color.alphaF());
      });
    }
  }
};

} // namespace pc::ui::qml