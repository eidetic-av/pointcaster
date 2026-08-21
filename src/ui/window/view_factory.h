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

  KDDockWidgets::Core::View *
  createTabBar(KDDockWidgets::Core::TabBar *tab_bar,
               KDDockWidgets::Core::View *parent) const override;

  KDDockWidgets::Core::ClassicIndicatorWindowViewInterface *
  createClassicIndicatorWindow(
      KDDockWidgets::Core::ClassicDropIndicatorOverlay *drop_indicator_overlay,
      KDDockWidgets::Core::View *parent) const override;
};

} // namespace pc::ui
