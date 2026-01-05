#include "workspace.h"
#include <Corrade/PluginManager/Manager.h>
#include <QCoreApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QUrl>
#include <fstream>
#include <thread>
#include <kddockwidgets/Config.h>
#include <kddockwidgets/qtquick/Platform.h>
#include <memory>
#include <models/config_adapter.h>
#include <models/workspace_model.h>
#include <print>
#include <qcoreapplication.h>
#include "ui/window/custom_titlebar.h"

using namespace pc;

int main(int argc, char *argv[]) {
  QGuiApplication app(argc, argv);

  qmlRegisterUncreatableType<ConfigAdapter>(
      "Pointcaster", 1, 0, "ConfigAdapter",
      "ConfigAdapter is an abstract base type");

  QQmlApplicationEngine engine;

  KDDockWidgets::initFrontend(KDDockWidgets::FrontendType::QtQuick);
  KDDockWidgets::QtQuick::Platform::instance()->setQmlEngine(&engine);
  auto &docking_config = KDDockWidgets::Config::self();
  docking_config.setViewFactory(new pc::ui::CustomTitlebarViewFactory());

  WorkspaceConfiguration workspace_config;


  // workspace_config.devices.push_back(pc::devices::OrbbecDeviceConfiguration{
  //     .id = "test", .ip = "192.168.1.107"});
  //     save_workspace_to_file(workspace_config, "workspace.json");


  Workspace workspace(workspace_config);

  ui::WorkspaceModel workspace_model{workspace, &app,
                                     [] { QCoreApplication::exit(); }};

  // could do _workspace.load_config_from_file here
  // if its set to load on startup

  engine.rootContext()->setContextProperty("workspaceModel", &workspace_model);

  QObject::connect(
      &engine, &QQmlApplicationEngine::objectCreationFailed, &app,
      []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

  engine.loadFromModule("Pointcaster.Workspace", "MainWindow");


  // pc::load_workspace(workspace_config, "workspace.json");

  // std::jthread loader_thread([&]() {
  //   using namespace std::chrono_literals;
  //   std::this_thread::sleep_for(5s);

  //   // Push result to GUI thread and move it into the master config
  //   QMetaObject::invokeMethod(
  //       &workspace_model,
  //       [&, cfg = std::move(loaded_config)]() mutable {
  //         workspace_config = std::move(cfg);
  //         workspace.revert_config();
  //         workspace_model.reloadFromWorkspace();
  //       },
  //       Qt::QueuedConnection);
  // });


  return app.exec();
}
