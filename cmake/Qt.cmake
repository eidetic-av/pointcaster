# use the system qt installation, not vcpkg

set(QT_NO_PRIVATE_MODULE_WARNING ON)
set(QT_QML_GENERATE_QMLLS_INI ON)

find_package(Qt6 REQUIRED COMPONENTS
    Core Gui Qml Quick QuickControls2 QuickDialogs2
    OpenGL Quick3D GuiPrivate QuickPrivate
)

qt_policy(SET QTP0004 NEW)

find_package(KDDockWidgets-qt6 CONFIG REQUIRED)
