#include "plugins/devices/device_plugin.h"

#include "plugins/devices/device_variants.h"
#include "plugins/devices/orbbec/orbbec_device_config.gen.h"

#include "plugins/devices/orbbec/orbbec_device_config.h"
#include "workspace.h"
#include <Corrade/PluginManager/Manager.h>
#include <QCoreApplication>

#include <QGuiApplication>

#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QUrl>
#include <kddockwidgets/Config.h>
#include <kddockwidgets/qtquick/Platform.h>
#include <kddockwidgets/qtquick/ViewFactory.h>
#include <models/config_adapter.h>
#include <models/workspace_model.h>
#include <print>

class CustomTitlebarViewFactory : public KDDockWidgets::QtQuick::ViewFactory {
public:
  ~CustomTitlebarViewFactory() override;

  QUrl titleBarFilename() const override {
    return QUrl(QStringLiteral(
        "qrc:/qt/qml/Pointcaster/Workspace/Window/TitleBar.qml"));
  }
};

CustomTitlebarViewFactory::~CustomTitlebarViewFactory() = default;

int main(int argc, char *argv[]) {
  QGuiApplication app(argc, argv);

  qmlRegisterUncreatableType<ConfigAdapter>(
      "Pointcaster", 1, 0, "ConfigAdapter",
      "ConfigAdapter is an abstract base type");

  QQmlApplicationEngine engine;

  KDDockWidgets::initFrontend(KDDockWidgets::FrontendType::QtQuick);
  KDDockWidgets::QtQuick::Platform::instance()->setQmlEngine(&engine);
  auto &docking_config = KDDockWidgets::Config::self();
  docking_config.setViewFactory(new CustomTitlebarViewFactory());

  ////////////

  pc::WorkspaceConfiguration workspace_config;
  pc::Workspace workspace;

  pc::ui::WorkspaceModel workspace_model{workspace, &app};
  engine.rootContext()->setContextProperty("workspaceModel", &workspace_model);

  ///////////

  using DevicePluginManager =
      Corrade::PluginManager::Manager<pc::devices::DevicePlugin>;

  std::unique_ptr<DevicePluginManager> device_plugin_manager;
  device_plugin_manager = std::make_unique<
      Corrade::PluginManager::Manager<pc::devices::DevicePlugin>>();

  workspace_config.devices.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "test",
                                             .active = true,
                                             .ip = "192.168.1.107",
                                             .depth_mode = 0,
                                             .acquisition_mode = 0});
  workspace_config.devices.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "abc",
                                             .active = true,
                                             .ip = "192.168.1.109",
                                             .depth_mode = 0,
                                             .acquisition_mode = 0});

  const auto loaded_orbbec_plugin =
      (device_plugin_manager->load("OrbbecDevice") &
       Corrade::PluginManager::LoadState::Loaded);

  /////////////

  workspace.devices.reserve(workspace_config.devices.size());

  for (auto &device_config_variant : workspace_config.devices) {
    std::visit(
        [&](auto &device_config) {
          using ConfigVariant = std::decay_t<decltype(device_config)>;

          if constexpr (std::is_same_v<
                            ConfigVariant,
                            pc::devices::OrbbecDeviceConfiguration>) {

            // add the config to our workspace model for ui
            workspace_model.addOrbbecDeviceAdapter(device_config);

            if (loaded_orbbec_plugin) {
              Corrade::Containers::Pointer<pc::devices::DevicePlugin>
                  orbbec_plugin;

              orbbec_plugin =
                  device_plugin_manager->instantiate("OrbbecDevice");
              orbbec_plugin->set_config(device_config_variant);

              // add it to our workspace
              workspace.devices.push_back(std::move(orbbec_plugin));

            } else {
              device_config.active = false;
            }
          }
        },
        device_config_variant);
  }

  ////////////

  QObject::connect(
      &engine, &QQmlApplicationEngine::objectCreationFailed, &app,
      []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

  engine.loadFromModule("Pointcaster.Workspace", "MainWindow");

  return app.exec();
}
