#pragma once

#include <optional>
#include <string>

class QQmlApplicationEngine;
class QGuiApplication;

namespace pc {
class Workspace;
namespace ui {
class WorkspaceModel;
}
} // namespace pc

namespace pc::ui {

QQmlApplicationEngine *
initialise(QGuiApplication *app, pc::ui::WorkspaceModel *workspace_model,
           const std::optional<std::string> &loaded_workspace_path);

void load_main_window(Workspace *workspace, QGuiApplication *app,
                      QQmlApplicationEngine *engine);
} // namespace pc::ui