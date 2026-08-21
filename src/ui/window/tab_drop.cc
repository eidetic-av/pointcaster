#include "tab_drop.h"
#include "tab_bar_view.h"
#include <QQuickItem>
#include <algorithm>
#include <kddockwidgets/core/Draggable_p.h>
#include <kddockwidgets/core/Group.h>
#include <kddockwidgets/core/Platform.h>
#include <kddockwidgets/core/TabBar.h>
#include <kddockwidgets/qtquick/TabBar.h>

namespace pc::ui {

TabDropTarget &TabDropTarget::instance() {
  static TabDropTarget target;
  return target;
}

void TabDropTarget::clear() {
  group = nullptr;
  insert_index = -1;
  dragged_dock_widgets.clear();
}

QRectF tab_bar_global_rect(KDDockWidgets::Core::TabBar *tab_bar) {
  if (!tab_bar)
    return {};

  auto *const tab_bar_view =
      dynamic_cast<KDDockWidgets::QtQuick::TabBar *>(tab_bar->view());
  if (!tab_bar_view)
    return {};

  QQuickItem *const tab_bar_item = tab_bar_view->tabBarQmlItem();
  if (!tab_bar_item || !tab_bar_item->isVisible() ||
      tab_bar_item->height() <= 0)
    return {};

  return QRectF(tab_bar_item->mapToGlobal(QPointF(0, 0)),
                QSizeF(tab_bar_item->width(), tab_bar_item->height()));
}

TabInsertPoint tab_insert_point_at(KDDockWidgets::Core::TabBar *tab_bar,
                                   QPointF global_position) {
  TabInsertPoint insert_point;

  auto *const tab_bar_view =
      dynamic_cast<KDDockWidgets::QtQuick::TabBar *>(tab_bar->view());
  if (!tab_bar_view)
    return insert_point;

  const auto tab_count = tab_bar->numDockWidgets();
  insert_point.index = tab_count;
  insert_point.global_caret_x = tab_bar_global_rect(tab_bar).left();

  for (auto i = 0; i < tab_count; i++) {
    const QRect tab_rect = tab_bar_view->globalRectForTab(i);
    if (global_position.x() < tab_rect.center().x()) {
      insert_point.index = i;
      insert_point.global_caret_x = tab_rect.left();
      return insert_point;
    }
    insert_point.global_caret_x = tab_rect.right();
  }

  return insert_point;
}

bool tab_drop_indicator_allowed(
    KDDockWidgets::DropLocation location,
    const QVector<KDDockWidgets::Core::DockWidget *> &source,
    const QVector<KDDockWidgets::Core::DockWidget *> &,
    KDDockWidgets::Core::DropArea *) {
  // kddw only reaches here for a centre drop once the hovered group's
  // affinities accept the dragged window, so this doubles as our affinity gate
  if (location == KDDockWidgets::DropLocation_Center)
    TabDropTarget::instance().dragged_dock_widgets = source;
  return true;
}

bool allow_drag_to_start(KDDockWidgets::Core::Draggable *draggable) {
  auto *const tab_bar = dynamic_cast<KDDockWidgets::Core::TabBar *>(draggable);
  if (!tab_bar)
    return true;

  const QRectF tab_bar_rect = tab_bar_global_rect(tab_bar);
  if (tab_bar_rect.isEmpty())
    return true;

  // hold the press inside the strip so tabs can be reordered along it;
  // leaving the strip is what breaks the window away
  const QPoint cursor_position =
      KDDockWidgets::Core::Platform::instance()->cursorPos();
  if (tab_bar_rect.contains(cursor_position))
    return false;

  if (auto *const tab_bar_view = dynamic_cast<TabBarView *>(tab_bar->view()))
    tab_bar_view->cancel_reorder();

  return true;
}

void apply_pending_tab_drop() {
  auto &target = TabDropTarget::instance();

  auto *const group = target.group.data();
  const auto insert_index = target.insert_index;
  const auto dragged_dock_widgets = target.dragged_dock_widgets;
  target.clear();

  if (!group || insert_index < 0 || dragged_dock_widgets.isEmpty())
    return;

  auto *const tab_bar = group->tabBar();
  if (!tab_bar)
    return;

  // kddw appended the dropped widgets, so walk them back to the drop point
  auto destination = insert_index;
  for (auto *dock_widget : dragged_dock_widgets) {
    const auto current_index = group->indexOfDockWidget(dock_widget);
    if (current_index == -1)
      continue;
    const auto clamped_destination =
        std::clamp(destination, 0, tab_bar->numDockWidgets() - 1);
    if (current_index != clamped_destination)
      tab_bar->moveTabTo(current_index, clamped_destination);
    destination = clamped_destination + 1;
  }
}

} // namespace pc::ui
