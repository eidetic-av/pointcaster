qt_standard_project_setup(REQUIRES 6.11)

include(GNUInstallDirs)

set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_INSTALL_QMLDIR "qml")
set(QT_QML_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/qml")

add_subdirectory(ui)

qt_add_executable(pointcaster
    pointcaster.cc
)

target_link_libraries(pointcaster PRIVATE ${UI_LINK_LIBS})

qt_import_qml_plugins(pointcaster)

install(TARGETS pointcaster
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
