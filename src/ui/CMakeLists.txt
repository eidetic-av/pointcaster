add_library(pointcaster_ui
    models/workspace_model.cc
)
add_library(pointcaster::ui ALIAS pointcaster_ui)

target_include_directories(pointcaster_ui PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(qml/Pointcaster/Workspace)

find_package(Corrade CONFIG REQUIRED REQUIRED Main Utility)
target_link_libraries(pointcaster_ui PRIVATE
    pointcaster::core Corrade::Main Corrade::Utility
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::OpenGL
    Qt6::Quick Qt6::QuickControls2 Qt6::Quick3D
    KDAB::kddockwidgets
)

target_link_libraries(pointcaster PRIVATE 
    pointcaster::core pointcaster::ui
    Corrade::Main Corrade::Utility
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::OpenGL
    Qt6::Quick Qt6::QuickControls2 Qt6::Quick3D
    KDAB::kddockwidgets
    PointcasterWorkspace
)

qt_import_qml_plugins(pointcaster)