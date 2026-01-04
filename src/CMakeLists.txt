qt_standard_project_setup(REQUIRES 6.11)

include(GNUInstallDirs)

set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_INSTALL_QMLDIR "qml")
set(QT_QML_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/qml")

qt_add_executable(pointcaster 
    pointcaster.cc
)

add_subdirectory(ui)

# corrade lib used for dynamic plugin / lib loading
find_package(Corrade CONFIG REQUIRED Main PluginManager)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)
target_link_libraries(pointcaster PRIVATE Corrade::Main Corrade::PluginManager)

# platform, device & operator plugins
add_subdirectory(plugins)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_custom_command(TARGET pointcaster POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    -DROOT=$<TARGET_FILE_DIR:pointcaster>
    -P ${CMAKE_SOURCE_DIR}/cmake/PruneMsvcDependencies.cmake
  )
endif()

install(TARGETS pointcaster
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
