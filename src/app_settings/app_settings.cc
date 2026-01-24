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
  m_restoreLastSession = m_settings.value("restoreLastSession", true).toBool();
  m_lastSessionPath = m_settings.value("lastSessionPath", "").toString();

  // Store as string in QSettings, e.g. "debug", "info", ...
  const auto logLevelText =
      m_settings.value("logLevel", QStringLiteral("info")).toString();
  m_logLevel = logLevelFromString(logLevelText);

  m_uiScale = m_settings.value("ui/scale", 1.0).toDouble();

  m_enablePrometheusMetrics =
      m_settings.value("metrics/enabled", true).toBool();
  m_prometheusAddress =
      m_settings
          .value("metrics/prometheusAddress", QStringLiteral("0.0.0.0:8080"))
          .toString();
}

AppSettings::LogLevel AppSettings::logLevel() const {
  return m_logLevel;
}

void AppSettings::setLogLevel(LogLevel level) {
  if (level == m_logLevel) return;
  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, level] { setLogLevel(level); }, Qt::QueuedConnection);
    return;
  }
  m_logLevel = level;
  write("logLevel", logLevelToString(m_logLevel));
  emit logLevelChanged();
}

AppSettings::LogLevel AppSettings::logLevelFromString(QStringView levelText) {
  const QString s = levelText.trimmed().toString().toLower();

  if (s == QLatin1String("trace")) return LogLevel::Trace;
  if (s == QLatin1String("debug")) return LogLevel::Debug;
  if (s == QLatin1String("info")) return LogLevel::Info;

  if (s == QLatin1String("warn") || s == QLatin1String("warning"))
    return LogLevel::Warn;

  if (s == QLatin1String("error") || s == QLatin1String("err"))
    return LogLevel::Error;

  if (s == QLatin1String("critical") || s == QLatin1String("crit"))
    return LogLevel::Critical;

  if (s == QLatin1String("off")) return LogLevel::Off;

  return LogLevel::Info;
}

QString AppSettings::logLevelToString(LogLevel level) {
  switch (level) {
  case LogLevel::Trace:
    return QStringLiteral("trace");
  case LogLevel::Debug:
    return QStringLiteral("debug");
  case LogLevel::Info:
    return QStringLiteral("info");
  case LogLevel::Warn:
    return QStringLiteral("warn");
  case LogLevel::Error:
    return QStringLiteral("error");
  case LogLevel::Critical:
    return QStringLiteral("critical");
  case LogLevel::Off:
    return QStringLiteral("off");
  }
  return QStringLiteral("info");
}

spdlog::level::level_enum AppSettings::spdlogLogLevel() const {
  return static_cast<spdlog::level::level_enum>(static_cast<int>(m_logLevel));
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
  write("metrics/enabled", m_enablePrometheusMetrics);
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
