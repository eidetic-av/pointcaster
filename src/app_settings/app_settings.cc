#include "app_settings.h"

#include <QCoreApplication>
#include <QMetaObject>
#include <QThread>
#include <QtGlobal>
#include <qcoreapplication.h>

namespace pc {

static bool onObjectThread(QObject *obj) {
  return QThread::currentThread() == obj->thread();
}

AppSettings *AppSettings::instance() {
  QCoreApplication::setOrganizationName("matth");
  QCoreApplication::setApplicationName("pointcaster");
  static AppSettings *s_instance = new AppSettings(qApp);
  return s_instance;
}

AppSettings::AppSettings(QObject *parent)
    : QObject(parent), m_settings(QSettings::NativeFormat, QSettings::UserScope,
                                  QCoreApplication::organizationName(),
                                  QCoreApplication::applicationName()) {
  // Load initial cache
  m_restoreLastSession =
      m_settings.value("restoreLastSession", true).toBool();
  m_lastSessionPath =
      m_settings.value("lastSessionPath", "").toString();

  m_uiScale = m_settings.value("ui/scale", 1.0).toDouble();

  m_enablePrometheusMetrics =
      m_settings.value("metrics/enabled", true).toBool();
  m_prometheusAddress =
      m_settings
          .value("metrics/prometheusAddress", QStringLiteral("0.0.0.0:8080"))
          .toString();
}

double AppSettings::uiScale() const {
  return m_uiScale;
}

void AppSettings::setUiScale(double value) {
  const double v = qBound(0.5, value, 3.0);
  if (qFuzzyCompare(v, m_uiScale)) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, v] { setUiScale(v); }, Qt::QueuedConnection);
    return;
  }

  m_uiScale = v;
  write("ui/scale", m_uiScale);
  emit uiScaleChanged();
}

bool AppSettings::restoreLastSession() const {
  return m_restoreLastSession;
}

void AppSettings::setRestoreLastSession(bool value) {
  if (value == m_restoreLastSession) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, value] { setRestoreLastSession(value); },
        Qt::QueuedConnection);
    return;
  }

  m_restoreLastSession = value;
  write("restoreLastSession", m_restoreLastSession);
  emit restoreLastSessionChanged();
}

QString AppSettings::lastSessionPath() const {
  return m_lastSessionPath;
}

void AppSettings::setLastSessionPath(const QString &value) {
  if (value == m_lastSessionPath) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, value] { setLastSessionPath(value); },
        Qt::QueuedConnection);
    return;
  }

  m_lastSessionPath = value;
  write("lastSessionPath", m_lastSessionPath);
  emit lastSessionPathChanged();
}


bool AppSettings::enablePrometheusMetrics() const {
  return m_enablePrometheusMetrics;
}

void AppSettings::setEnablePrometheusMetrics(bool value) {
  if (value == m_enablePrometheusMetrics) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, value] { setEnablePrometheusMetrics(value); },
        Qt::QueuedConnection);
    return;
  }

  m_enablePrometheusMetrics = value;
  write("metrics", m_enablePrometheusMetrics);
  emit enablePrometheusMetricsChanged();
}

QString AppSettings::prometheusAddress() const {
  return m_prometheusAddress;
}

void AppSettings::setPrometheusAddress(const QString &value) {
  if (value == m_prometheusAddress) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, value] { setPrometheusAddress(value); },
        Qt::QueuedConnection);
    return;
  }

  m_prometheusAddress = value;
  write("metrics/prometheusAddress", m_prometheusAddress);
  emit prometheusAddressChanged();
}

QVariant AppSettings::value(const QString &key,
                            const QVariant &defaultValue) const {
  return m_settings.value(key, defaultValue);
}

void AppSettings::setValue(const QString &key, const QVariant &value) {
  if (key.isEmpty()) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, key, value] { setValue(key, value); },
        Qt::QueuedConnection);
    return;
  }

  write(key, value);
  emit valueChanged(key);
}

void AppSettings::sync() {
  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(this, [this] { sync(); }, Qt::QueuedConnection);
    return;
  }
  m_settings.sync();
}

void AppSettings::write(const QString &key, const QVariant &v) {
  m_settings.setValue(key, v);
}

} // namespace pc