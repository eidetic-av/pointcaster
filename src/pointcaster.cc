#include "workspace.h"
#include <QCoreApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <app_settings/app_settings.h>
#include <core/logger/logger.h>
#include <print>
#include <ui/initialisation.h>

#include <rfl.hpp>
#include <rfl/AddTagsToVariants.hpp>
#include <rfl/json.hpp>

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

  pc::logger->debug("Starting QGuiApplication...");
  QGuiApplication app(argc, argv);

  pc::logger->trace("Starting QQmlApplicationEngine...");
  auto *gui_engine = pc::ui::initialise(&app);

  pc::logger->trace("Initialising Workspace...");

  WorkspaceConfiguration workspace_config;
  if (app_settings->restoreLastSession() &&
      !app_settings->lastSessionPath().isEmpty()) {
    load_workspace_from_file(workspace_config,
                             app_settings->lastSessionPath().toStdString());
  }

  Workspace workspace(workspace_config);

  // TODO
  // quick: every 1 second, update workspace->set config
  // to the one thats been altered by the gui if its been marked dirty?
  // or maybe we mark individual parameters as dirty, yeah...

  pc::logger->trace("Loading main window...");

  pc::ui::load_main_window(&workspace, &app, gui_engine);

  pc::logger->trace("Application load complete");

  return app.exec();
}
