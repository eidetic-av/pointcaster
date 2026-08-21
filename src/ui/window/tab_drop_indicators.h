#pragma once

#include <QPointer>
#include <QQuickItem>
#include <QRectF>
#include <kddockwidgets/qtquick/ClassicIndicatorsWindow.h>

namespace KDDockWidgets::Core {
class ClassicDropIndicatorOverlay;
class View;
} // namespace KDDockWidgets::Core

namespace pc::ui {

// extends kddw's classic drop indicators with insertion points along the tab
// strip of whichever group is hovered, so a dragged dock window can be tabbed
// at a chosen position instead of always being appended
class TabDropIndicators
    : public KDDockWidgets::QtQuick::ClassicDropIndicatorOverlay {
  Q_OBJECT
public:
  TabDropIndicators(
      KDDockWidgets::Core::ClassicDropIndicatorOverlay *drop_indicator_overlay,
      KDDockWidgets::Core::View *parent);

  KDDockWidgets::DropLocation hover(QPoint global_position) override;
  void setVisible(bool visible) override;

private:
  void ensure_indicator_tiles();
  void apply_indicator_scale();
  KDDockWidgets::DropLocation hover_tab_bar(QPoint global_position);
  void show_tab_strip_overlay(QRectF global_tab_bar_rect, qreal global_caret_x);
  void hide_tab_strip_overlay();
  QQuickItem *tab_strip_overlay();

  KDDockWidgets::Core::ClassicDropIndicatorOverlay *const m_drop_indicator_overlay;
  QPointer<QQuickItem> m_tab_strip_overlay;
  bool m_indicator_tiles_installed = false;
};

} // namespace pc::ui
