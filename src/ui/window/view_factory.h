#pragma once

#include <QUrl>
#include <kddockwidgets/qtquick/ViewFactory.h>

namespace pc::ui {

class CustomViewFactory : public KDDockWidgets::QtQuick::ViewFactory {
public:
  ~CustomViewFactory() override;
  QUrl tabbarFilename() const override;
  QUrl separatorFilename() const override;
  QUrl floatingWindowFilename() const override;
  QUrl groupFilename() const override;
};

} // namespace pc::ui