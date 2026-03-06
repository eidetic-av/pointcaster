#include <QCoreApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <app_settings/app_settings.h>
#include <core/logger/logger.h>
#include <optional>
#include <print>
#include <ui/initialisation.h>
#include <ui/models/workspace_model.h>
#include <workspace/workspace.h>

using namespace pc;

int main(int argc, char *argv[]) {

  std::println("This is pointcaster 0.2.0");

  auto *app_settings = AppSettings::instance();
  qmlRegisterSingletonInstance("Pointcaster", 1, 0, "AppSettings",
                               app_settings);

  pc::set_log_level(app_settings->spdlogLogLevel());
  QObject::connect(
      app_settings, &pc::AppSettings::logLevelChanged, app_settings,
      [] { pc::set_log_level(pc::AppSettings::instance()->spdlogLogLevel()); });

  pc::logger()->debug("Starting QGuiApplication...");
  QGuiApplication app(argc, argv);

  pc::logger()->trace("Initialising Workspace...");

  WorkspaceConfiguration workspace_config;

  std::optional<std::string> loaded_workspace_path;
  if (app_settings->restoreLastWorkspace() &&
      !app_settings->lastWorkspacePath().isEmpty()) {
    const auto last_workspace_path =
        app_settings->lastWorkspacePath().toStdString();
    if (load_workspace_from_file(workspace_config, last_workspace_path)) {
      loaded_workspace_path = last_workspace_path;
    };
  }

  // create a pointcaster workspace using this initial configuration, either
  // auto-loaded or empty
  Workspace workspace(workspace_config);

  pc::ui::WorkspaceModel workspace_model{&workspace, &app};

  pc::logger()->trace("Starting QQmlApplicationEngine...");
  auto *gui_engine = pc::ui::initialise(&app, &workspace_model, loaded_workspace_path);

  pc::ui::load_main_window(&workspace, &app, gui_engine);

  pc::logger()->trace("Application load complete");

  return app.exec();
}
