
# singletons inside qml must be specified to cmake before
# configuring the qml module

set(qml_singletons
    WorkspaceState.qml
    Theme/DarkPalette.qml
    Theme/FontAwesome.qml
    Utils/StringUtils.qml
)
foreach(qml_singleton_file IN LISTS qml_singletons)
    set_source_files_properties( ${qml_singleton_file} 
        PROPERTIES QT_QML_SINGLETON_TYPE TRUE)
endforeach()

qt_add_qml_module(PointcasterWorkspace 
	URI Pointcaster.Workspace VERSION 1.0
    QML_FILES
        ${qml_singletons}
        Controls/DragInput.qml
        Controls/DragInt.qml
        Controls/DragFloat.qml
        Controls/DragFloat3.qml
        Controls/EnumSelector.qml
        Controls/IconButton.qml
        Editors/ConfigurationEditor.qml
        Windows/TitleBar.qml
        Windows/TabBar.qml
        Windows/TabBarBase.qml
        Windows/Separator.qml
        Windows/FloatingWindow.qml
        Windows/Group.qml
        Windows/ResizeHandlerHelper.qml
        Windows/TopMenuBar.qml
        Windows/MainWindow.qml
        Windows/Devices/DevicesWindow.qml
        Windows/Devices/DeviceSelectionList.qml
        Windows/Sessions/SessionView.qml
        Windows/Sessions/SessionViewWindow.qml
        Windows/Sessions/CentralSessionView.qml
    RESOURCES
        Resources/Fonts/AtkinsonHyperlegibleNext-Light.otf
        Resources/Fonts/AtkinsonHyperlegibleNext-Regular.otf
        Resources/Fonts/AtkinsonHyperlegibleNext-Medium.otf
        Resources/Fonts/AtkinsonHyperlegibleNext-Bold.otf
)

find_package(QtAwesome CONFIG REQUIRED)

target_link_libraries(PointcasterWorkspace PRIVATE
    pointcaster::core
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::OpenGL
    Qt6::Quick Qt6::QuickControls2 Qt6::QuickDialogs2
    Qt6::Quick3D
    KDAB::kddockwidgets
    QtAwesome::QtAwesome
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
