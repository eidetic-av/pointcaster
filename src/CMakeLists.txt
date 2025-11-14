qt_policy(SET QTP0004 NEW)

qt_standard_project_setup(REQUIRES 6.8)

include(GNUInstallDirs)

set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_INSTALL_QMLDIR "qml")
set(QT_QML_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/qml")

qt_add_qml_module(PointcasterControls
	URI Pointcaster.Controls
	VERSION 1.0
	QML_FILES
		ui/qml/Pointcaster/Controls/TextButton.qml
)
install(DIRECTORY "${QT_QML_OUTPUT_DIRECTORY}/Pointcaster/Controls" 
	DESTINATION "${CMAKE_INSTALL_QMLDIR}/Pointcaster")

qt_add_qml_module(PointcasterApp
	URI Pointcaster.App
	VERSION 1.0
	QML_FILES
		ui/qml/Pointcaster/App/App.qml
)
install(DIRECTORY "${QT_QML_OUTPUT_DIRECTORY}/Pointcaster/App"
	DESTINATION "${CMAKE_INSTALL_QMLDIR}/Pointcaster")

# qt_add_executable(pointcaster pointcaster.cc)
qt_add_executable(pointcaster ${CMAKE_CURRENT_SOURCE_DIR}/../customtitlebar/main.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../customtitlebar/resources_qtquick_example.qrc
	/opt/KDDockWidgets/examples/dockwidgets/resources_example.qrc
)

# TODO make platform specific
#qt_import_plugins(pointcaster INCLUDE
#  Qt6::QWaylandIntegrationPlugin)

#set_target_properties(pointcaster PROPERTIES
#WIN32_EXECUTABLE TRUE
#)

target_link_libraries(pointcaster
	PRIVATE PointcasterGUI
		PointcasterControls
		PointcasterApp
)

install(TARGETS pointcaster
	BUNDLE DESTINATION .
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS 
	PointcasterGUI
	PointcasterControls
	PointcasterApp
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

add_custom_target(linuxdeploy
    COMMAND "${CMAKE_SOURCE_DIR}/packaging/install-with-linuxdeploy.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
