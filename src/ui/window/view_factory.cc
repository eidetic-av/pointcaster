#include "view_factory.h"
#include "tab_bar_view.h"
#include "tab_drop_indicators.h"
#include <QStringLiteral>
#include <kddockwidgets/qtquick/View.h>

namespace pc::ui {

CustomViewFactory::~CustomViewFactory() = default;

QUrl CustomViewFactory::separatorFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Windows/Separator.qml"));
}

QUrl CustomViewFactory::tabbarFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Windows/TabBar.qml"));
}

QUrl CustomViewFactory::floatingWindowFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Windows/FloatingWindow.qml"));
}

QUrl CustomViewFactory::groupFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Windows/Group.qml"));
}

KDDockWidgets::Core::View *
CustomViewFactory::createTabBar(KDDockWidgets::Core::TabBar *tab_bar,
                                KDDockWidgets::Core::View *parent) const {
  return new TabBarView(tab_bar, KDDockWidgets::QtQuick::asQQuickItem(parent));
}

KDDockWidgets::Core::ClassicIndicatorWindowViewInterface *
CustomViewFactory::createClassicIndicatorWindow(
    KDDockWidgets::Core::ClassicDropIndicatorOverlay *drop_indicator_overlay,
    KDDockWidgets::Core::View *parent) const {
  return new TabDropIndicators(drop_indicator_overlay, parent);
}

} // namespace pc::ui
