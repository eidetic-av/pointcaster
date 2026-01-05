qt_add_qml_module(PointcasterWorkspace 
	URI Pointcaster.Workspace VERSION 1.0
    QML_FILES
        Windows/TitleBar.qml
        Windows/MainWindow.qml
        Windows/DevicesWindow.qml
        Windows/SessionViewWindow.qml
        Controls/TextButton.qml
    RESOURCES
        Resources/Fonts/AtkinsonHyperlegibleNext-Regular.otf
)

target_link_libraries(PointcasterWorkspace PRIVATE
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::OpenGL
    Qt6::Quick Qt6::QuickControls2 Qt6::QuickDialogs2
    Qt6::Quick3D
    KDAB::kddockwidgets
)

install(DIRECTORY "${QT_QML_OUTPUT_DIRECTORY}/Pointcaster/Workspace"
	DESTINATION "${CMAKE_INSTALL_QMLDIR}/Pointcaster"
)

install(TARGETS PointcasterWorkspace
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
