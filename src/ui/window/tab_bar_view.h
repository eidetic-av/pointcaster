#pragma once

#include <QPointF>
#include <kddockwidgets/qtquick/TabBar.h>

namespace pc::ui {

// kddw's qtquick TabBar leaves moveTabTo() unimplemented, so a reorder updates
// the controller's list while the qml model keeps the old order. it also has no
// counterpart to QTabBar's movable tabs, so dragging along the strip is handled
// here rather than being left to break the window away
class TabBarView : public KDDockWidgets::QtQuick::TabBar {
  Q_OBJECT
public:
  explicit TabBarView(KDDockWidgets::Core::TabBar *tab_bar,
                      QQuickItem *parent = nullptr);

  void moveTabTo(int from, int to) override;
  void setCurrentIndex(int index) override;

  // detach a tab into its own floating window
  Q_INVOKABLE void floatTabAt(int index);

  // kddw consumes the mouse move that starts a drag, so the break-away has to
  // tell us to drop the reorder feedback rather than us noticing it ourselves
  void cancel_reorder();

protected:
  bool event(QEvent *event) override;

private:
  void update_reorder(QPointF local_position);
  void finish_reorder();
  void clear_reorder();
  void publish_reorder();

  bool m_moving_tab = false;
  int m_pressed_tab_index = -1;
  int m_reorder_insert_index = -1;
  QPointF m_press_position;
};

} // namespace pc::ui
