#include "custom_titlebar.h"
#include <QStringLiteral>

namespace pc::ui {

CustomTitlebarViewFactory::~CustomTitlebarViewFactory() = default;

QUrl CustomTitlebarViewFactory::titleBarFilename() const {
  return QUrl(
      QStringLiteral("qrc:/qt/qml/Pointcaster/Workspace/Window/TitleBar.qml"));
}

} // namespace pc::ui