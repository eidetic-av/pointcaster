#include "tab_bar_view.h"
#include "tab_drop.h"
#include <QMouseEvent>
#include <QQuickItem>
#include <algorithm>
#include <kddockwidgets/core/DockWidget.h>
#include <kddockwidgets/core/Platform.h>
#include <kddockwidgets/core/TabBar.h>

namespace pc::ui {

TabBarView::TabBarView(KDDockWidgets::Core::TabBar *tab_bar, QQuickItem *parent)
    : KDDockWidgets::QtQuick::TabBar(tab_bar, parent) {}

bool TabBarView::event(QEvent *event) {
  switch (event->type()) {
  case QEvent::MouseButtonPress: {
    auto *const mouse_event = static_cast<QMouseEvent *>(event);
    if (mouse_event->button() == Qt::LeftButton) {
      m_press_position = mouse_event->position();
      m_pressed_tab_index = tabAt(m_press_position.toPoint());
    }
    break;
  }
  case QEvent::MouseMove: {
    auto *const mouse_event = static_cast<QMouseEvent *>(event);
    if (mouse_event->buttons() & Qt::LeftButton)
      update_reorder(mouse_event->position());
    break;
  }
  case QEvent::MouseButtonRelease:
    finish_reorder();
    break;
  default:
    break;
  }

  // the base swallows presses and warns that it may have been deleted, so all
  // of our own bookkeeping has to happen first
  return KDDockWidgets::QtQuick::TabBar::event(event);
}

void TabBarView::update_reorder(QPointF local_position) {
  if (m_pressed_tab_index == -1)
    return;

  QQuickItem *const tab_bar_item = tabBarQmlItem();
  if (!tab_bar_item)
    return;

  const QRectF tab_bar_rect(0, 0, tab_bar_item->width(),
                            tab_bar_item->height());
  if (!tab_bar_rect.contains(local_position)) {
    // out of the strip, so kddw is free to break the window away from here on
    clear_reorder();
    return;
  }

  const auto start_drag_distance =
      KDDockWidgets::Core::Platform::instance()->startDragDistance();
  const auto drag_distance =
      (local_position - m_press_position).manhattanLength();
  if (drag_distance < start_drag_distance)
    return;

  const auto insert_point =
      tab_insert_point_at(m_tabBar, tab_bar_item->mapToGlobal(local_position));
  if (insert_point.index == m_reorder_insert_index)
    return;

  m_reorder_insert_index = insert_point.index;
  publish_reorder();
}

void TabBarView::finish_reorder() {
  const auto from = m_pressed_tab_index;
  const auto insert_index = m_reorder_insert_index;

  clear_reorder();
  m_pressed_tab_index = -1;

  if (from == -1 || insert_index == -1)
    return;

  // the insertion point counts the dragged tab, which is gone once it lands
  const auto landing_index = insert_index > from ? insert_index - 1 : insert_index;
  const auto destination =
      std::clamp(landing_index, 0, m_tabBar->numDockWidgets() - 1);
  if (destination != from)
    m_tabBar->moveTabTo(from, destination);
}

void TabBarView::cancel_reorder() {
  clear_reorder();
  m_pressed_tab_index = -1;
}

void TabBarView::clear_reorder() {
  if (m_reorder_insert_index == -1)
    return;
  m_reorder_insert_index = -1;
  publish_reorder();
}

void TabBarView::publish_reorder() {
  QQuickItem *const tab_bar_item = tabBarQmlItem();
  if (!tab_bar_item)
    return;

  const auto reorder_tab_index =
      m_reorder_insert_index == -1 ? -1 : m_pressed_tab_index;
  tab_bar_item->setProperty("reorderTabIndex", reorder_tab_index);
  tab_bar_item->setProperty("reorderInsertIndex", m_reorder_insert_index);
}

void TabBarView::moveTabTo(int from, int to) {
  auto *const model = dockWidgetModel();
  if (!model)
    return;

  // the controller reorders its own list before telling us, so the model still
  // holds the pre-move order here
  auto *const dock_widget = model->dockWidgetAt(from);
  if (!dock_widget)
    return;

  m_moving_tab = true;
  model->remove(dock_widget);
  model->insert(dock_widget, to);
  m_moving_tab = false;
}

void TabBarView::floatTabAt(int index) {
  auto *const model = dockWidgetModel();
  if (!model)
    return;

  auto *const dock_widget = model->dockWidgetAt(index);
  if (!dock_widget)
    return;

  dock_widget->setFloating(true);
}

void TabBarView::setCurrentIndex(int index) {
  // between the remove and the insert the model is a row short, and the tab bar
  // qml echoes the controller's index back to us against that shorter model
  if (m_moving_tab)
    return;
  KDDockWidgets::QtQuick::TabBar::setCurrentIndex(index);
}

} // namespace pc::ui
