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
#include <kddockwidgets/Config.h>
#include <kddockwidgets/core/DockRegistry.h>
#include <kddockwidgets/core/FloatingWindow.h>
#include <kddockwidgets/core/TitleBar.h>
#include <kddockwidgets/core/views/MainWindowViewInterface.h>
#include <kddockwidgets/qtquick/Platform.h>
#include <print>

namespace pc::ui {

QQmlApplicationEngine *initialise(QGuiApplication *app) {
  pc::ui::register_qml_uncreatable_types();

  static QQmlApplicationEngine engine;

  KDDockWidgets::initFrontend(KDDockWidgets::FrontendType::QtQuick);
  KDDockWidgets::QtQuick::Platform::instance()->setQmlEngine(&engine);
  auto &docking_config = KDDockWidgets::Config::self();
  docking_config.setViewFactory(new pc::ui::CustomViewFactory());

  static fa::QtAwesome awesome(app);
  awesome.initFontAwesome();
  engine.addImageProvider("fa", new QtAwesomeQuickImageProvider(&awesome));

  return &engine;
}

void load_main_window(Workspace *workspace, QGuiApplication *app,
                      QQmlApplicationEngine *engine) {

  static ui::WorkspaceModel workspace_model{workspace, app,
                                            [] { QCoreApplication::exit(); }};

  // could do _workspace.load_config_from_file here
  // if its set to load on startup

  engine->rootContext()->setContextProperty("workspaceModel", &workspace_model);

  QObject::connect(
      engine, &QQmlApplicationEngine::objectCreationFailed, app,
      []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

  // ---- QML dev override: load a local file from disk if provided ----
  const QString dev_main_qml = qEnvironmentVariable("POINTCASTER_MAIN_QML");
  const QString dev_qml_import_root =
      qEnvironmentVariable("POINTCASTER_QML_IMPORT_ROOT");

  if (!dev_qml_import_root.isEmpty()) {
    engine->addImportPath(dev_qml_import_root);
  }

  if (!dev_main_qml.isEmpty()) {
    // ensure imports work even if only POINTCASTER_MAIN_QML is set
    // by also adding the directory containing the file as an import path.
    const QFileInfo mainQmlInfo(dev_main_qml);
    if (mainQmlInfo.exists()) {
      engine->addImportPath(mainQmlInfo.dir().absolutePath());
      engine->load(QUrl::fromLocalFile(mainQmlInfo.absoluteFilePath()));
    } else {
      std::println("POINTCASTER_MAIN_QML points to missing file: {}",
                   dev_main_qml.toStdString());
      QCoreApplication::exit(-1);
    }
  } else {
    // normal packaged/built module path
    engine->loadFromModule("Pointcaster.Workspace", "MainWindow");
  }
}

} // namespace pc::ui
