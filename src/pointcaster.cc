#include "workspace.h"
#include <app_settings/app_settings.h>
#include <QCoreApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <ui/initialisation.h>
#include <print>

using namespace pc;

int main(int argc, char *argv[]) {
  std::println("Starting QGuiApplication...");
  QGuiApplication app(argc, argv);

  std::println("Loading application preferences...");
  auto* app_settings = AppSettings::instance();
  qmlRegisterSingletonInstance("Pointcaster", 1, 0, "AppSettings", app_settings);

  std::println("Starting QQmlApplicationEngine...");
  auto* gui_engine = pc::ui::initialise(&app);

  std::println("Initialising Workspace...");

  WorkspaceConfiguration workspace_config;
  if (app_settings->restoreLastSession() && !app_settings->lastSessionPath().isEmpty()) {
    load_workspace_from_file(workspace_config,
                             app_settings->lastSessionPath().toStdString());
  }
//   workspace_config.devices.push_back(pc::devices::OrbbecDeviceConfiguration{
//       .id = "test", .ip = "192.168.1.107"});
//   workspace_config.devices.push_back(pc::devices::OrbbecDeviceConfiguration{
//       .id = "test2", .ip = "192.168.1.108"});

  // for (int i = 0; i < 6; i++) {
  //   workspace_config.devices.push_back(pc::devices::OrbbecDeviceConfiguration{
  //       .id = std::format("test{}", i),
  //       .ip = std::format("192.168.1.10{}", 8 + i)});
  // }
  // save_workspace_to_file(workspace_config, "workspace.json");

  Workspace workspace(workspace_config);
  
  std::println("Loading main window...");

  pc::ui::load_main_window(&workspace, &app, gui_engine);

  std::println("Application load complete");

  return app.exec();
}
