add_subdirectory(qml/Pointcaster/Workspace)

target_link_libraries(pointcaster PRIVATE 
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::OpenGL
    Qt6::Quick Qt6::QuickControls2 Qt6::Quick3D
    KDAB::kddockwidgets
    PointcasterWorkspace
)

qt_import_qml_plugins(pointcaster)