add_library(pointcaster_ui
    initialisation.cc
    models/workspace_model.cc
    models/device_status.h
    models/settings_page_registry.h
    models/settings_page_registry.cc
    models/commands/device_config_commands.cc
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

# for our generated headers that connect configuration structs <-> qml
set(CONFIG_ADAPTER_INPUTS
    ${CMAKE_SOURCE_DIR}/src/session/session_config.h
    ${CMAKE_SOURCE_DIR}/src/camera/camera_config.h
    ${CMAKE_SOURCE_DIR}/src/plugins/devices/orbbec/orbbec_device_config.h
)
add_qml_adapters(
    TARGETS pointcaster_ui
    CONFIG_INPUTS ${CONFIG_ADAPTER_INPUTS}
    TEMPLATES_DIR "${CMAKE_SOURCE_DIR}/src/templates"
    OUT_DIR "${CMAKE_BINARY_DIR}/generated/config_adapters"
)

target_precompile_headers(pointcaster_ui PRIVATE
  <string>
  <string_view>
  <vector>
  <memory>
  <mutex>
  <optional>
  <QObject>
  <QString>
  <QVariant>
  <QUrl>
  <QCoreApplication>
  <QThread>
  <QMetaObject>
  <QUndoCommand>
)