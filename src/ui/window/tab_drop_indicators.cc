#include "tab_drop_indicators.h"
#include "tab_drop.h"
#include <QQmlContext>
#include <QQmlEngine>
#include <QQuickView>
#include <QStringLiteral>
#include <QUrl>
#include <app_settings/app_settings.h>
#include <array>
#include <kddockwidgets/core/Group.h>
#include <kddockwidgets/core/TabBar.h>
#include <kddockwidgets/core/indicators/ClassicDropIndicatorOverlay.h>
#include <kddockwidgets/qtquick/View.h>

namespace pc::ui {

namespace {

QString tab_strip_overlay_qml() {
  return QStringLiteral("qrc:/qt/qml/Pointcaster/Windows/TabDropIndicator.qml");
}

QString drop_indicator_tile_qml() {
  return QStringLiteral("qrc:/qt/qml/Pointcaster/Windows/DropIndicatorTile.qml");
}

// the indicator window builds its own engine, which has no reason to know about
// our module, so anything of ours loaded into it can reach ThemeColors
void ensure_pointcaster_imports(QQmlEngine *engine) {
  static const QString module_import_path = QStringLiteral("qrc:/qt/qml");
  if (!engine->importPathList().contains(module_import_path))
    engine->addImportPath(module_import_path);
}

struct IndicatorAnchor {
  KDDockWidgets::DropLocation location;
  const char *margin_property;
};

// kddw's overlay qml spaces each indicator off its neighbour or the window edge
// with a fixed margin, so those have to follow the tile size
constexpr std::array<IndicatorAnchor, 9> indicator_anchors = {{
    {KDDockWidgets::DropLocation_Center, nullptr},
    {KDDockWidgets::DropLocation_Left, "rightMargin"},
    {KDDockWidgets::DropLocation_Right, "leftMargin"},
    {KDDockWidgets::DropLocation_Top, "bottomMargin"},
    {KDDockWidgets::DropLocation_Bottom, "topMargin"},
    {KDDockWidgets::DropLocation_OutterLeft, "leftMargin"},
    {KDDockWidgets::DropLocation_OutterRight, "rightMargin"},
    {KDDockWidgets::DropLocation_OutterTop, "topMargin"},
    {KDDockWidgets::DropLocation_OutterBottom, "bottomMargin"},
}};

} // namespace

TabDropIndicators::TabDropIndicators(
    KDDockWidgets::Core::ClassicDropIndicatorOverlay *drop_indicator_overlay,
    KDDockWidgets::Core::View *parent)
    : KDDockWidgets::QtQuick::ClassicDropIndicatorOverlay(drop_indicator_overlay,
                                                          parent),
      m_drop_indicator_overlay(drop_indicator_overlay) {}

KDDockWidgets::DropLocation TabDropIndicators::hover(QPoint global_position) {
  ensure_indicator_tiles();

  // the classic indicators win any overlap, so existing docking is untouched
  const auto classic_location =
      KDDockWidgets::QtQuick::ClassicDropIndicatorOverlay::hover(
          global_position);
  if (classic_location != KDDockWidgets::DropLocation_None) {
    hide_tab_strip_overlay();
    TabDropTarget::instance().clear();
    return classic_location;
  }

  const auto tab_location = hover_tab_bar(global_position);
  if (tab_location == KDDockWidgets::DropLocation_None) {
    hide_tab_strip_overlay();
    TabDropTarget::instance().clear();
  }
  return tab_location;
}

void TabDropIndicators::setVisible(bool visible) {
  if (visible)
    ensure_indicator_tiles();

  if (!visible) {
    hide_tab_strip_overlay();
    TabDropTarget::instance().clear();
  }
  KDDockWidgets::QtQuick::ClassicDropIndicatorOverlay::setVisible(visible);
}

void TabDropIndicators::ensure_indicator_tiles() {
  if (m_indicator_tiles_installed)
    return;

  QQuickItem *const centre_indicator =
      indicatorForLocation(KDDockWidgets::DropLocation_Center);
  if (!centre_indicator)
    return;

  QQmlEngine *const engine = qmlEngine(centre_indicator);
  if (!engine)
    return;

  ensure_pointcaster_imports(engine);

  for (const auto &[location, margin_property] : indicator_anchors) {
    QQuickItem *const indicator = indicatorForLocation(location);
    if (!indicator)
      continue;

    // drops the binding onto the stock bitmap, the tile draws everything now
    indicator->setProperty("source", QUrl());

    QQuickItem *const tile = KDDockWidgets::QtQuick::View::createItem(
        engine, drop_indicator_tile_qml(), engine->rootContext());
    if (!tile)
      continue;

    tile->setParentItem(indicator);
    tile->setParent(indicator);
    KDDockWidgets::QtQuick::View::makeItemFillParent(tile);
  }

  connect(AppSettings::instance(), &AppSettings::uiScaleChanged, this,
          &TabDropIndicators::apply_indicator_scale);

  m_indicator_tiles_installed = true;
  apply_indicator_scale();
}

void TabDropIndicators::apply_indicator_scale() {
  const qreal ui_scale = AppSettings::instance()->uiScale();
  const qreal tile_size = qRound(40.0 * ui_scale);
  const qreal indicator_margin = qRound(10.0 * ui_scale);

  for (const auto &[location, margin_property] : indicator_anchors) {
    QQuickItem *const indicator = indicatorForLocation(location);
    if (!indicator)
      continue;

    indicator->setWidth(tile_size);
    indicator->setHeight(tile_size);

    if (!margin_property)
      continue;

    auto *const anchors = indicator->property("anchors").value<QObject *>();
    if (anchors)
      anchors->setProperty(margin_property, indicator_margin);
  }
}

KDDockWidgets::DropLocation
TabDropIndicators::hover_tab_bar(QPoint global_position) {
  using namespace KDDockWidgets;

  auto *const group = m_drop_indicator_overlay->hoveredGroup();
  if (!group)
    return DropLocation_None;

  // this runs kddw's own centre-drop checks, affinities included
  if (!m_drop_indicator_overlay->dropIndicatorVisible(DropLocation_Center))
    return DropLocation_None;

  auto *const tab_bar = group->tabBar();
  if (!tab_bar)
    return DropLocation_None;

  const QRectF tab_bar_rect = tab_bar_global_rect(tab_bar);
  if (!tab_bar_rect.contains(global_position))
    return DropLocation_None;

  const auto insert_point = tab_insert_point_at(tab_bar, global_position);
  if (insert_point.index == -1)
    return DropLocation_None;

  auto &target = TabDropTarget::instance();
  target.group = group;
  target.insert_index = insert_point.index;

  show_tab_strip_overlay(tab_bar_rect, insert_point.global_caret_x);

  return DropLocation_Center;
}

QQuickItem *TabDropIndicators::tab_strip_overlay() {
  if (m_tab_strip_overlay)
    return m_tab_strip_overlay;

  QQuickItem *const centre_indicator =
      indicatorForLocation(KDDockWidgets::DropLocation_Center);
  if (!centre_indicator)
    return nullptr;

  auto *const indicator_window =
      qobject_cast<QQuickView *>(centre_indicator->window());
  if (!indicator_window)
    return nullptr;

  ensure_pointcaster_imports(indicator_window->engine());

  QQuickItem *const overlay = KDDockWidgets::QtQuick::View::createItem(
      indicator_window->engine(), tab_strip_overlay_qml(),
      indicator_window->rootContext());
  if (!overlay)
    return nullptr;

  overlay->setParentItem(indicator_window->contentItem());
  overlay->setParent(indicator_window->contentItem());
  overlay->setZ(100);
  overlay->setVisible(false);

  m_tab_strip_overlay = overlay;
  return overlay;
}

void TabDropIndicators::show_tab_strip_overlay(QRectF global_tab_bar_rect,
                                               qreal global_caret_x) {
  QQuickItem *const overlay = tab_strip_overlay();
  if (!overlay)
    return;

  QQuickItem *const container = overlay->parentItem();
  if (!container)
    return;

  const QPointF local_origin =
      container->mapFromGlobal(global_tab_bar_rect.topLeft());

  overlay->setPosition(local_origin);
  overlay->setSize(global_tab_bar_rect.size());
  overlay->setProperty("caretX", global_caret_x - global_tab_bar_rect.left());
  overlay->setVisible(true);
}

void TabDropIndicators::hide_tab_strip_overlay() {
  if (m_tab_strip_overlay)
    m_tab_strip_overlay->setVisible(false);
}

} // namespace pc::ui
