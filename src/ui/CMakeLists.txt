add_library(pointcaster_ui
    initialisation.cc
    models/workspace_model.cc
    models/device_status.h
    models/device_plugin_controller.h
    models/settings_page_registry.h
    models/settings_page_registry.cc
    window/view_factory.cc
)
add_library(pointcaster::ui ALIAS pointcaster_ui)

target_include_directories(pointcaster_ui PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(qml/Pointcaster/Workspace)

find_package(Corrade CONFIG REQUIRED REQUIRED Main Utility)
find_package(QtAwesome CONFIG REQUIRED)

target_link_libraries(pointcaster_ui PRIVATE
    pointcaster::core 
    pointcaster::app_settings
    Corrade::Main Corrade::Utility
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::OpenGL
    Qt6::Quick Qt6::QuickControls2 Qt6::Quick3D
    KDAB::kddockwidgets 
    PointcasterWorkspace
    QtAwesome::QtAwesome
)

target_link_libraries(pointcaster PRIVATE Qt6::Gui)

qt_import_qml_plugins(pointcaster)