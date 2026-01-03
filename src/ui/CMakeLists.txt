add_subdirectory(qml/Pointcaster/Workspace)

set(UI_LINK_LIBS
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::OpenGL
	Qt6::Quick Qt6::QuickControls2 Qt6::Quick3D
    KDAB::kddockwidgets
    PointcasterWorkspace
    PARENT_SCOPE
)