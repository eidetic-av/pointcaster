#include "color_picker.h"

namespace pc::ui::qml {

void ColorPicker::pick(QQuickItem *target, int x, int y) {
  if (target) {
    pc::logger()->trace("ColorPicker.pick(x: {}, y: {}", x, y);
    const auto grab_result = target->grabToImage();
    QObject::connect(grab_result.data(), &QQuickItemGrabResult::ready,
                     [=, this] {
                       pc::logger()->trace("grabToImage completed");
                       auto image = grab_result->image();
                       qreal dpr = image.devicePixelRatio();
                       int pixel_x = static_cast<int>(x * dpr);
                       int pixel_y = static_cast<int>(y * dpr);
                       QRgb pixel = image.pixel(pixel_x, pixel_y);
                       _r = qRed(pixel);
                       _g = qGreen(pixel);
                       _b = qBlue(pixel);
                       _a = qAlpha(pixel);
                       emit picked();
                     });
  }
}

} // namespace pc::ui::qml