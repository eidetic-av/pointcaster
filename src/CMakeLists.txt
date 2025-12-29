qt_standard_project_setup(REQUIRES 6.11)

include(GNUInstallDirs)

set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_INSTALL_QMLDIR "qml")
set(QT_QML_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/qml")

qt_add_executable(pointcaster pointcaster.cc)

add_subdirectory(ui)

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
