#include "workspace.h"
#include <QCoreApplication>
#include <QGuiApplication>
#include <ui/initialisation.h>

using namespace pc;

int main(int argc, char *argv[]) {
  QGuiApplication app(argc, argv);

  auto* gui_engine = pc::ui::initialise(&app);

  WorkspaceConfiguration workspace_config;
  workspace_config.devices.push_back(pc::devices::OrbbecDeviceConfiguration{
      .id = "test", .ip = "192.168.1.107"});
  workspace_config.devices.push_back(pc::devices::OrbbecDeviceConfiguration{
      .id = "test2", .ip = "192.168.1.108"});

  // for (int i = 0; i < 6; i++) {
  //   workspace_config.devices.push_back(pc::devices::OrbbecDeviceConfiguration{
  //       .id = std::format("test{}", i),
  //       .ip = std::format("192.168.1.10{}", 8 + i)});
  // }
  // save_workspace_to_file(workspace_config, "workspace.json");

  Workspace workspace(workspace_config);

  pc::ui::load_main_window(&workspace, &app, gui_engine);

  return app.exec();
}
