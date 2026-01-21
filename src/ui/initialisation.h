#pragma once

class QQmlApplicationEngine;
class QGuiApplication;

namespace pc {
    class Workspace;
}

namespace pc::ui {

    QQmlApplicationEngine* initialise(QGuiApplication* app);

    void load_main_window(Workspace* workspace, QGuiApplication* app, QQmlApplicationEngine* engine);
}