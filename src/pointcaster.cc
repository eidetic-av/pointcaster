#include <QCoreApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QUrl>
#include <kddockwidgets/Config.h>
#include <kddockwidgets/qtquick/Platform.h>
#include <kddockwidgets/qtquick/ViewFactory.h>

#include <Corrade/PluginManager/Manager.h>
#include "plugins/devices/orbbec/orbbec_device.h"

#include <cpplocate/cpplocate.h>
#include <filesystem>

#include <iostream>

class CustomTitlebarViewFactory : public KDDockWidgets::QtQuick::ViewFactory {
public:
    ~CustomTitlebarViewFactory() override;

  QUrl titleBarFilename() const override {
    return QUrl(
        QStringLiteral("qrc:/qt/qml/Pointcaster/Workspace/Window/TitleBar.qml"));
    }
};

CustomTitlebarViewFactory::~CustomTitlebarViewFactory() = default;

int main(int argc, char *argv[])
{
	QGuiApplication app(argc, argv);

	QQmlApplicationEngine engine;

	KDDockWidgets::initFrontend(KDDockWidgets::FrontendType::QtQuick);
	KDDockWidgets::QtQuick::Platform::instance()->setQmlEngine(&engine);
	auto &docking_config = KDDockWidgets::Config::self();
	docking_config.setViewFactory(new CustomTitlebarViewFactory());

	QObject::connect(&engine,
			&QQmlApplicationEngine::objectCreationFailed,
			&app,
			[]() { QCoreApplication::exit(-1); },
			Qt::QueuedConnection);

  	engine.loadFromModule("Pointcaster.Workspace", "MainWindow");

	const auto exe_path = cpplocate::getExecutablePath();
	const auto exe_dir = std::filesystem::path(exe_path).parent_path();
	const auto plugin_dir = exe_dir.parent_path() / "plugins";

  using OrbbecPluginManager = Corrade::PluginManager::Manager<pc::devices::AbstractOrbbecDevice>;
  std::unique_ptr<OrbbecPluginManager> _orbbec_plugin_manager;
  Corrade::Containers::Pointer<pc::devices::AbstractOrbbecDevice> _orbbec_plugin;

//   std::atomic_bool _orbbec_tried_load = false;
//   std::atomic_bool _orbbec_loaded = false;
//   std::jthread _plugin_load_thread;

    _orbbec_plugin_manager = std::make_unique<Corrade::PluginManager::Manager<
        pc::devices::AbstractOrbbecDevice>>();


	/*const auto plugin_path = (plugin_dir / "OrbbecDevice.so").string();*/
	const auto plugin_path = (plugin_dir / "orbbec" / "OrbbecDevice.dll").string();
    std::cout << "plugin path is going to be: " << plugin_path << "\n";
	/*const auto loaded_orbbec_plugin =*/
	/*	(_orbbec_plugin_manager->load(plugin_path) & Corrade::PluginManager::LoadState::Loaded);*/

	const auto loaded_orbbec_plugin =
		(_orbbec_plugin_manager->load("OrbbecDevice") & Corrade::PluginManager::LoadState::Loaded);

	if (loaded_orbbec_plugin) {
		std::cout << "oi!\n";
	}
	
    // if ( &
    //     Corrade::PluginManager::LoadState::Loaded) {
    //   _orbbec_plugin = _orbbec_plugin_manager->instantiate("OrbbecDevice");
    // }

	return app.exec();
}
