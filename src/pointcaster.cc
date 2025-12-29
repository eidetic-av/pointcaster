#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <kddockwidgets/Config.h>
#include <kddockwidgets/qtquick/Platform.h>
#include <kddockwidgets/qtquick/ViewFactory.h>

class CustomTitlebarViewFactory : public KDDockWidgets::QtQuick::ViewFactory
{
public:
    ~CustomTitlebarViewFactory() override;

    QUrl titleBarFilename() const override
    {
	    return QUrl::fromLocalFile(
			    QCoreApplication::applicationDirPath() +
			    "/../qml/Pointcaster/Controls/qml/Pointcaster/Controls/TitleBar.qml");
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

	engine.loadFromModule("Pointcaster.App", "App");

	return app.exec();
}
