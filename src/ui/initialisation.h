#pragma once

#include <optional>
#include <string>

class QQmlApplicationEngine;
class QGuiApplication;

namespace pc {
class Workspace;
}

namespace pc::ui {

QQmlApplicationEngine *
initialise(QGuiApplication *app,
           const std::optional<std::string> &loaded_session_path);

void load_main_window(Workspace *workspace, QGuiApplication *app,
                      QQmlApplicationEngine *engine);
} // namespace pc::ui