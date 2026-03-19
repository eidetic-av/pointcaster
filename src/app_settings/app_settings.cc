#include "app_settings.h"

#include <QCoreApplication>
#include <QMetaObject>
#include <QThread>
#include <QtGlobal>
#include <QtQml/qqml.h>
#include <qcoreapplication.h>

namespace pc {

static bool onObjectThread(QObject *obj) {
  return QThread::currentThread() == obj->thread();
}

AppSettings *AppSettings::instance() {
  QCoreApplication::setOrganizationName("matth");
  QCoreApplication::setApplicationName("pointcaster");
  static AppSettings *s_instance = new AppSettings(qApp);
  qmlRegisterSingletonInstance("Pointcaster", 1, 0, "AppSettings", s_instance);
  return s_instance;
}

AppSettings::AppSettings(QObject *parent)
    : QObject(parent), m_settings(QSettings::NativeFormat, QSettings::UserScope,
                                  QCoreApplication::organizationName(),
                                  QCoreApplication::applicationName()) {
  // Load initial cache
  m_restoreLastWorkspace =
      m_settings.value("restoreLastWorkspace", true).toBool();
  m_lastWorkspacePath = m_settings.value("lastWorkspacePath", "").toString();

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

  m_enableTracyProfiling = m_settings.value("profiling/enabled", true).toBool();
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

bool AppSettings::logToFile() const {
  return m_logToFile;
}

void AppSettings::setLogToFile(bool value) {
  if (value == m_logToFile) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, value] { setLogToFile(value); }, Qt::QueuedConnection);
    return;
  }

  m_logToFile = value;
  write("logToFile", m_logToFile);
  emit logToFileChanged();
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

bool AppSettings::restoreLastWorkspace() const {
  return m_restoreLastWorkspace;
}

void AppSettings::setrestoreLastWorkspace(bool value) {
  if (value == m_restoreLastWorkspace) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, value] { setrestoreLastWorkspace(value); },
        Qt::QueuedConnection);
    return;
  }

  m_restoreLastWorkspace = value;
  write("restoreLastWorkspace", m_restoreLastWorkspace);
  emit restoreLastWorkspaceChanged();
}

QString AppSettings::lastWorkspacePath() const {
  return m_lastWorkspacePath;
}

void AppSettings::setlastWorkspacePath(const QString &value) {
  if (value == m_lastWorkspacePath) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, value] { setlastWorkspacePath(value); },
        Qt::QueuedConnection);
    return;
  }

  m_lastWorkspacePath = value;
  write("lastWorkspacePath", m_lastWorkspacePath);
  emit lastWorkspacePathChanged();
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

bool AppSettings::enableTracyProfiling() const {
  return m_enableTracyProfiling;
}

void AppSettings::setEnableTracyProfiling(bool value) {
  if (value == m_enableTracyProfiling) return;

  if (!onObjectThread(this)) {
    QMetaObject::invokeMethod(
        this, [this, value] { setEnableTracyProfiling(value); },
        Qt::QueuedConnection);
    return;
  }

  m_enableTracyProfiling = value;
  write("profiling/enabled", m_enableTracyProfiling);
  emit enableTracyProfilingChanged();
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
