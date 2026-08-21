#pragma once

#include <QPointF>
#include <QPointer>
#include <QRectF>
#include <QVector>
#include <kddockwidgets/KDDockWidgets.h>

namespace KDDockWidgets::Core {
class DockWidget;
class Draggable;
class DropArea;
class Group;
class TabBar;
} // namespace KDDockWidgets::Core

namespace pc::ui {

struct TabDropTarget {
  static TabDropTarget &instance();

  QPointer<KDDockWidgets::Core::Group> group;
  int insert_index = -1;
  QVector<KDDockWidgets::Core::DockWidget *> dragged_dock_widgets;

  void clear();
};

struct TabInsertPoint {
  int index = -1;
  qreal global_caret_x = 0;
};

QRectF tab_bar_global_rect(KDDockWidgets::Core::TabBar *tab_bar);

TabInsertPoint tab_insert_point_at(KDDockWidgets::Core::TabBar *tab_bar,
                                   QPointF global_position);

bool tab_drop_indicator_allowed(
    KDDockWidgets::DropLocation location,
    const QVector<KDDockWidgets::Core::DockWidget *> &source,
    const QVector<KDDockWidgets::Core::DockWidget *> &target,
    KDDockWidgets::Core::DropArea *drop_area);

bool allow_drag_to_start(KDDockWidgets::Core::Draggable *draggable);

void apply_pending_tab_drop();

} // namespace pc::ui
