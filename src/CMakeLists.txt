qt_standard_project_setup(REQUIRES 6.11)

include(GNUInstallDirs)

set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_INSTALL_QMLDIR "qml")
set(QT_QML_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/qml")

qt_add_executable(pointcaster 
    pointcaster.cc
)

add_subdirectory(ui)
qt_import_qml_plugins(pointcaster)

# libs needed for plugin loading
find_package(boost_dll)
find_package(Corrade CONFIG REQUIRED Main PluginManager)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)
target_link_libraries(pointcaster PRIVATE Corrade::Main Corrade::PluginManager Boost::dll)

# the plugins themselves
add_subdirectory(plugins)

# find_package(Corrade CONFIG REQUIRED Main)
# set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

# target_link_libraries(pointcaster PRIVATE 
#     Corrade::Main
# )

install(TARGETS pointcaster
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
