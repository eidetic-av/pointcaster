set_source_files_properties(Theme/DarkPalette.qml
    PROPERTIES QT_QML_SINGLETON_TYPE TRUE)

qt_add_qml_module(PointcasterWorkspace 
	URI Pointcaster.Workspace VERSION 1.0
    QML_FILES
        Theme/DarkPalette.qml
        Windows/TitleBar.qml
        Windows/MainWindow.qml
        Windows/Devices/DevicesWindow.qml
        Windows/Devices/DeviceSelectionList.qml
        Windows/Sessions/SessionView.qml
        Windows/Sessions/SessionViewWindow.qml
        Windows/Sessions/CentralSessionView.qml
        Controls/TextButton.qml
    RESOURCES
        Resources/Fonts/AtkinsonHyperlegibleNext-Light.otf
        Resources/Fonts/AtkinsonHyperlegibleNext-Regular.otf
        Resources/Fonts/AtkinsonHyperlegibleNext-Medium.otf
        Resources/Fonts/AtkinsonHyperlegibleNext-Bold.otf
)

target_link_libraries(PointcasterWorkspace PRIVATE
    pointcaster::core
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::OpenGL
    Qt6::Quick Qt6::QuickControls2 Qt6::QuickDialogs2
    Qt6::Quick3D
    KDAB::kddockwidgets
)

# for pointcaster::core headers in public include dir
target_include_directories(PointcasterWorkspace 
    PRIVATE "${CMAKE_SOURCE_DIR}/include")

install(DIRECTORY "${QT_QML_OUTPUT_DIRECTORY}/Pointcaster/Workspace"
	DESTINATION "${CMAKE_INSTALL_QMLDIR}/Pointcaster"
)

install(TARGETS PointcasterWorkspace
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
