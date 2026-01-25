#include "view_factory.h"
#include <QStringLiteral>

namespace pc::ui {

CustomViewFactory::~CustomViewFactory() = default;

QUrl CustomViewFactory::separatorFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Workspace/Windows/Separator.qml"));
}

QUrl CustomViewFactory::tabbarFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Workspace/Windows/TabBar.qml"));
}

QUrl CustomViewFactory::floatingWindowFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Workspace/Windows/FloatingWindow.qml"));
}

QUrl CustomViewFactory::groupFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Workspace/Windows/Group.qml"));
}

} // namespace pc::ui