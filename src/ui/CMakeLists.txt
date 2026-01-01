
# GUI deps

find_package(KDDockWidgets-qt6 CONFIG REQUIRED)

qt_import_qml_plugins(pointcaster)
qt_import_plugins(pointcaster INCLUDE kddockwidgetsplugin Qt6::Quick3D)

# the gui library

add_library(PointcasterGUI INTERFACE)
target_link_libraries(PointcasterGUI INTERFACE
	Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick
	Qt6::Quick3D Qt6::QuickControls2 Qt6::OpenGL
	KDAB::kddockwidgets
)

# add qml component libs

qt_add_qml_module(PointcasterControls
	URI Pointcaster.Controls
	VERSION 1.0
	QML_FILES
	qml/Pointcaster/Controls/TitleBar.qml
	qml/Pointcaster/Controls/TextButton.qml
)

target_link_libraries(PointcasterControls PRIVATE PointcasterGUI)

install(
	DIRECTORY "${QT_QML_OUTPUT_DIRECTORY}/Pointcaster/Controls"
	DESTINATION "${CMAKE_INSTALL_QMLDIR}/Pointcaster"
)

qt_add_qml_module(PointcasterApp
	URI Pointcaster.App
	VERSION 1.0
	QML_FILES
	qml/Pointcaster/App/App.qml
)

install(DIRECTORY "${QT_QML_OUTPUT_DIRECTORY}/Pointcaster/App"
	DESTINATION "${CMAKE_INSTALL_QMLDIR}/Pointcaster"
)

set(ATKINSON_FONT
    ${CMAKE_SOURCE_DIR}/resources/fonts/AtkinsonHyperlegibleNext-Regular.otf
)
set_source_files_properties(${ATKINSON_FONT}
    PROPERTIES
        QT_RESOURCE_ALIAS "fonts/AtkinsonHyperlegibleNext-Regular.otf"
)

qt_add_resources(PointcasterApp "fonts"
	PREFIX "/"
	FILES "${ATKINSON_FONT}"
)
