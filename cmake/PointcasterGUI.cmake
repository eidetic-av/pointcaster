include_guard(GLOBAL)

find_package(Qt6 REQUIRED COMPONENTS
	Core Gui Qml Quick QuickControls2 OpenGL
	# make platform deps generator expression	
	WaylandCompositor
)
find_package(KDDockWidgets-qt6 CONFIG REQUIRED)

add_library(PointcasterGUI INTERFACE)

target_link_libraries(PointcasterGUI INTERFACE
	Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick
	Qt6::QuickControls2 Qt6::OpenGL
  	KDAB::kddockwidgets
	# TODO platform dep should be generator expression
	Qt6::WaylandCompositor
)
