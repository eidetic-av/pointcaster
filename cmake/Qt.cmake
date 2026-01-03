
set(QT_NO_PRIVATE_MODULE_WARNING ON)

find_package(Qt6 REQUIRED COMPONENTS
        Core Gui Qml Quick QuickControls2 
	OpenGL Quick3D GuiPrivate QuickPrivate
)

qt_policy(SET QTP0004 NEW)

find_package(KDDockWidgets-qt6 CONFIG REQUIRED)
