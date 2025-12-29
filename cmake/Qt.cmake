
set(QT_NO_PRIVATE_MODULE_WARNING ON)

find_package(Qt6 REQUIRED COMPONENTS
        Core Gui Qml Quick QuickControls2 OpenGL
        GuiPrivate QuickPrivate
)

qt_policy(SET QTP0004 NEW)

