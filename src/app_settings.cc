#include "app_settings.h"

#include <QCoreApplication>
#include <QMetaObject>
#include <QThread>
#include <QtGlobal>

namespace pc {

static bool onObjectThread(QObject* obj)
{
    return QThread::currentThread() == obj->thread();
}

AppSettings* AppSettings::instance()
{
    static AppSettings* s_instance = new AppSettings(qApp);
    return s_instance;
}

AppSettings::AppSettings(QObject* parent)
    : QObject(parent)
    , m_settings(QSettings::NativeFormat, QSettings::UserScope,
                 QCoreApplication::organizationName(),
                 QCoreApplication::applicationName())
{
    // Load initial cache
    m_uiScale = m_settings.value("ui/scale", 1.0).toDouble();
    m_restoreLastSession = m_settings.value("app/restoreLastSession", true).toBool();
}

double AppSettings::uiScale() const { return m_uiScale; }

void AppSettings::setUiScale(double value)
{
    const double v = qBound(0.5, value, 3.0);
    if (qFuzzyCompare(v, m_uiScale))
        return;

    if (!onObjectThread(this)) {
        QMetaObject::invokeMethod(this, [this, v]{ setUiScale(v); }, Qt::QueuedConnection);
        return;
    }

    m_uiScale = v;
    write("ui/scale", m_uiScale);
    emit uiScaleChanged();
}

bool AppSettings::restoreLastSession() const { return m_restoreLastSession; }

void AppSettings::setrestoreLastSession(bool value)
{
    if (value == m_restoreLastSession)
        return;

    if (!onObjectThread(this)) {
        QMetaObject::invokeMethod(this, [this, value]{ setrestoreLastSession(value); }, Qt::QueuedConnection);
        return;
    }

    m_restoreLastSession = value;
    write("app/restoreLastSession", m_restoreLastSession);
    emit restoreLastSessionChanged();
}

QVariant AppSettings::value(const QString& key, const QVariant& defaultValue) const
{
    return m_settings.value(key, defaultValue);
}

void AppSettings::setValue(const QString& key, const QVariant& value)
{
    if (key.isEmpty())
        return;

    if (!onObjectThread(this)) {
        QMetaObject::invokeMethod(this, [this, key, value]{ setValue(key, value); }, Qt::QueuedConnection);
        return;
    }

    write(key, value);
    emit valueChanged(key);
}

void AppSettings::sync()
{
    if (!onObjectThread(this)) {
        QMetaObject::invokeMethod(this, [this]{ sync(); }, Qt::QueuedConnection);
        return;
    }
    m_settings.sync();
}

void AppSettings::write(const QString& key, const QVariant& v)
{
    m_settings.setValue(key, v);
}

}