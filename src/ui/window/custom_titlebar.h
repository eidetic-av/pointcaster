#pragma once

#include <QUrl>
#include <kddockwidgets/qtquick/ViewFactory.h>

namespace pc::ui {

class CustomTitlebarViewFactory : public KDDockWidgets::QtQuick::ViewFactory {
public:
  ~CustomTitlebarViewFactory() override;

  QUrl titleBarFilename() const override;
};

} // namespace pc::ui