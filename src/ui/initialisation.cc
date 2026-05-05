#include "initialisation.h"
#include "layout_saver.h"
#include "models/registrations.h"
#include "models/workspace_model.h"
#include "window/view_factory.h"
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QLibrary>
#include <QQmlContext>
#include <QQuickWindow>
#include <QSGRendererInterface>
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
#include <optional>
#include <print>
#include <string>

namespace pc::ui {

QQmlApplicationEngine *
initialise(QGuiApplication *app, pc::ui::WorkspaceModel *workspace_model,
           const std::optional<std::string> &loaded_workspace_path) {

  // use vulkan even on windows because D3D11/12 doesn't support arbitrary point
  // sizes in point-cloud shaders based that make use of vertex point rendering
  QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);

  // qml boilerplate
  pc::ui::register_qml_uncreatable_types();

  // load in any qml shared libs required before application engine starts
  QLibrary gizmo("gizmo3d");
  if (!gizmo.load())
    pc::logger()->error("Failed to load Gizmo3d QML library");
  else
    pc::logger()->trace("Loaded Gizmo3D QML library");

  // create single ApplicationEngine instance
  static QQmlApplicationEngine qml_engine;
  qml_engine.rootContext()->setContextProperty("workspaceModel",
                                               workspace_model);

  // initialise kddw
  KDDockWidgets::initFrontend(KDDockWidgets::FrontendType::QtQuick);
  KDDockWidgets::QtQuick::Platform::instance()->setQmlEngine(&qml_engine);
  auto &docking_config = KDDockWidgets::Config::self();
  docking_config.setViewFactory(new pc::ui::CustomViewFactory());
  using KDDockWidgets::Config;
  docking_config.setFlags(
      Config::Flag_HideTitleBarWhenTabsVisible | Config::Flag_AlwaysShowTabs |
      Config::Flag_AllowReorderTabs | Config::Flag_DisableDoubleClick);

  // initialise our font awesome singleton
  static fa::QtAwesome awesome(app);
  awesome.initFontAwesome();
  qml_engine.addImageProvider("fa", new QtAwesomeQuickImageProvider(&awesome));

  // if we auto-loaded a workspace, find any adjacent layout file to load the UI
  if (loaded_workspace_path.has_value()) {
    std::filesystem::path workspace_file_path{loaded_workspace_path.value()};
    auto layout_file_path = workspace_file_path;
    layout_file_path.replace_filename(workspace_file_path.stem().string() +
                                      "_layout.json");

    if (std::filesystem::exists(layout_file_path)) {

      // wait until we load all the windows before manipulating the layout
      QObject::connect(
          &qml_engine, &QQmlApplicationEngine::objectCreated, app,
          [fp = layout_file_path.string(), workspace_model]() {
            pc::ui::LayoutSaver saver(workspace_model);
            saver.load_file(fp.c_str());
          },
          Qt::QueuedConnection);
    }
  }

  // log to console if qml engine can't initialise
  QObject::connect(
      &qml_engine, &QQmlApplicationEngine::objectCreationFailed, app,
      []() {
        pc::logger()->error("Failed to initialise QML engine!");
        QCoreApplication::exit(-1);
      },
      Qt::QueuedConnection);

  return &qml_engine;
}

void load_main_window(Workspace *workspace, QGuiApplication *app,
                      QQmlApplicationEngine *engine) {
  pc::logger()->trace("Loading main window...");
  // ---- QML dev override: load a local file from disk if provided ----
  const QString dev_main_qml = qEnvironmentVariable("POINTCASTER_MAIN_QML");
  const QString dev_qml_import_root =
      qEnvironmentVariable("POINTCASTER_QML_IMPORT_ROOT");

  if (!dev_qml_import_root.isEmpty()) {
    engine->addImportPath(dev_qml_import_root);
  }

  if (!dev_main_qml.isEmpty()) {
    pc::logger()->trace("Loading QML from source...");
    // ensure imports work even if only POINTCASTER_MAIN_QML is set
    // by also adding the directory containing the file as an import path.
    const QFileInfo mainQmlInfo(dev_main_qml);
    if (mainQmlInfo.exists()) {
      engine->addImportPath(mainQmlInfo.dir().absolutePath());
      engine->load(QUrl::fromLocalFile(mainQmlInfo.absoluteFilePath()));
    } else {
      pc::logger()->error("POINTCASTER_MAIN_QML points to missing file: {}",
                          dev_main_qml.toStdString());
      QCoreApplication::exit(-1);
    }
  } else {
    pc::logger()->trace("Loading compiled QML...");
    // normal packaged/built module path
    try {
      engine->loadFromModule("Pointcaster", "MainWindow");
    } catch (...) {
      pc::logger()->error("Failed to load main window");
    }
  }
}

} // namespace pc::ui
