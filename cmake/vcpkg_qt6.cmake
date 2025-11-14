include_guard(GLOBAL)

find_package(Qt6 CONFIG REQUIRED COMPONENTS
	Core Gui Qml Quick QuickControls2 OpenGL)

add_library(gui_deps INTERFACE)

target_link_libraries(gui_deps INTERFACE
	Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick
	Qt6::QuickControls2 Qt6::OpenGL)

target_compile_features(gui_deps INTERFACE cxx_std_23)
