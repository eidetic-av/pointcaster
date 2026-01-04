#include <QCoreApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QUrl>
#include <kddockwidgets/Config.h>
#include <kddockwidgets/qtquick/Platform.h>
#include <kddockwidgets/qtquick/ViewFactory.h>

#include "plugins/devices/device_plugin.h"

#include "plugins/devices/device_variants.h"

#include "config_adapter.h"
#include "plugins/devices/orbbec/orbbec_device_config.gen.h"

#include <Corrade/PluginManager/Manager.h>

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

  using DevicePluginManager =
      Corrade::PluginManager::Manager<pc::devices::DevicePlugin>;

  std::unique_ptr<DevicePluginManager> device_plugin_manager;
  device_plugin_manager = std::make_unique<
      Corrade::PluginManager::Manager<pc::devices::DevicePlugin>>();

  std::vector<pc::devices::DeviceConfigurationVariant> device_configs;
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "test",
                                             .active = true,
                                             .ip = "192.168.1.107",
                                             .depth_mode = 0,
                                             .acquisition_mode = 0});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});
  device_configs.push_back(
      pc::devices::OrbbecDeviceConfiguration{.id = "one2",
                                             .active = true,
                                             .ip = "192.168.1.108",
                                             .depth_mode = 1,
                                             .acquisition_mode = 1});

  const auto loaded_orbbec_plugin =
      (device_plugin_manager->load("OrbbecDevice") &
       Corrade::PluginManager::LoadState::Loaded);

  Corrade::Containers::Pointer<pc::devices::DevicePlugin> orbbec_plugin;

  if (loaded_orbbec_plugin) {
    std::println("oi! loaded orbbec!");
    orbbec_plugin = device_plugin_manager->instantiate("OrbbecDevice");
  }

  /////////////

   using pc::devices::DeviceConfigurationVariant;
    using pc::devices::OrbbecDeviceConfiguration;
    using pc::devices::OrbbecDeviceConfigurationAdapter;

    // 2) QML-facing adapters (owning QObject side)
    QList<QObject*> device_config_adapters;
    device_config_adapters.reserve(static_cast<int>(device_configs.size()));

    for (auto &variant : device_configs) {
        std::visit(
            [&](auto &cfg) {
                using Cfg = std::decay_t<decltype(cfg)>;

                // for now, just handle the types you actually have
                if constexpr (std::is_same_v<Cfg, OrbbecDeviceConfiguration>) {
                    auto *adapter =
                        new OrbbecDeviceConfigurationAdapter(cfg, &app);
                    device_config_adapters.append(adapter);
                }
                // when you add more config types to DeviceConfigurationVariant,
                // extend this if constexpr chain with their adapters
            },
            variant
        );
    }

    // 3) Expose the list of QObject* adapters to QML
    engine.rootContext()->setContextProperty(
        "deviceConfigAdapters", QVariant::fromValue(device_config_adapters));

    ////////////

    QObject::connect(
        &engine, &QQmlApplicationEngine::objectCreationFailed, &app,
        []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

    engine.loadFromModule("Pointcaster.Workspace", "MainWindow");

    return app.exec();
}
