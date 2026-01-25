#include "initialisation.h"
#include "models/registrations.h"
#include "models/workspace_model.h"
#include "window/view_factory.h"
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QQmlContext>
#include <QUrl>
#include <QtAwesome/QtAwesome.h>
#include <QtAwesome/QtAwesomeQuickImageProvider.h>
#include <core/logger/logger.h>
#include <kddockwidgets/Config.h>
#include <kddockwidgets/core/DockRegistry.h>
#include <kddockwidgets/core/FloatingWindow.h>
#include <kddockwidgets/core/TitleBar.h>
#include <kddockwidgets/core/views/MainWindowViewInterface.h>
#include <kddockwidgets/qtquick/Platform.h>
#include <print>

namespace pc::ui {

QQmlApplicationEngine *initialise(QGuiApplication *app) {

  // qml boilerplate
  pc::ui::register_qml_uncreatable_types();

  // create single ApplicationEngine instance
  static QQmlApplicationEngine engine;

  // initialise kddw
  KDDockWidgets::initFrontend(KDDockWidgets::FrontendType::QtQuick);
  KDDockWidgets::QtQuick::Platform::instance()->setQmlEngine(&engine);
  auto &docking_config = KDDockWidgets::Config::self();
  docking_config.setViewFactory(new pc::ui::CustomViewFactory());
  using KDDockWidgets::Config;
  docking_config.setFlags(
      Config::Flag_HideTitleBarWhenTabsVisible | Config::Flag_AlwaysShowTabs |
      Config::Flag_AllowReorderTabs | Config::Flag_DisableDoubleClick);

  // initialise our font awesome singleton
  static fa::QtAwesome awesome(app);
  awesome.initFontAwesome();
  engine.addImageProvider("fa", new QtAwesomeQuickImageProvider(&awesome));

  return &engine;
}

void load_main_window(Workspace *workspace, QGuiApplication *app,
                      QQmlApplicationEngine *engine) {
  pc::logger->trace("Loading main window...");
  // ---- QML dev override: load a local file from disk if provided ----
  const QString dev_main_qml = qEnvironmentVariable("POINTCASTER_MAIN_QML");
  const QString dev_qml_import_root =
      qEnvironmentVariable("POINTCASTER_QML_IMPORT_ROOT");

  if (!dev_qml_import_root.isEmpty()) {
    engine->addImportPath(dev_qml_import_root);
  }

  if (!dev_main_qml.isEmpty()) {
    pc::logger->trace("Loading QML from source...");
    // ensure imports work even if only POINTCASTER_MAIN_QML is set
    // by also adding the directory containing the file as an import path.
    const QFileInfo mainQmlInfo(dev_main_qml);
    if (mainQmlInfo.exists()) {
      engine->addImportPath(mainQmlInfo.dir().absolutePath());
      engine->load(QUrl::fromLocalFile(mainQmlInfo.absoluteFilePath()));
    } else {
      pc::logger->error("POINTCASTER_MAIN_QML points to missing file: {}",
                        dev_main_qml.toStdString());
      QCoreApplication::exit(-1);
    }
  } else {
    pc::logger->trace("Loading compiled QML...");
    // normal packaged/built module path
    engine->loadFromModule("Pointcaster.Workspace", "MainWindow");
  }
}

} // namespace pc::ui
