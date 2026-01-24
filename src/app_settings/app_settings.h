// app_settings.h
#pragma once

#include <QObject>
#include <QSettings>
#include <QString>
#include <QVariant>
#include <spdlog/common.h>

namespace pc {

class AppSettings final : public QObject {
  Q_OBJECT

  Q_PROPERTY(bool restoreLastSession READ restoreLastSession WRITE
                 setRestoreLastSession NOTIFY restoreLastSessionChanged)
  Q_PROPERTY(QString lastSessionPath READ lastSessionPath WRITE
                 setLastSessionPath NOTIFY lastSessionPathChanged)

  Q_PROPERTY(
      LogLevel logLevel READ logLevel WRITE setLogLevel NOTIFY logLevelChanged)

  Q_PROPERTY(double uiScale READ uiScale WRITE setUiScale NOTIFY uiScaleChanged)

  Q_PROPERTY(
      bool enablePrometheusMetrics READ enablePrometheusMetrics WRITE
          setEnablePrometheusMetrics NOTIFY enablePrometheusMetricsChanged)

  Q_PROPERTY(QString prometheusAddress READ prometheusAddress WRITE
                 setPrometheusAddress NOTIFY prometheusAddressChanged)

public:
  static AppSettings *instance();

  explicit AppSettings(QObject *parent = nullptr);

  // -- General

  bool restoreLastSession() const;
  void setRestoreLastSession(bool value);

  QString lastSessionPath() const;
  void setLastSessionPath(const QString &value);

  enum class LogLevel : int {
    Trace = static_cast<int>(spdlog::level::trace),
    Debug = static_cast<int>(spdlog::level::debug),
    Info = static_cast<int>(spdlog::level::info),
    Warn = static_cast<int>(spdlog::level::warn),
    Error = static_cast<int>(spdlog::level::err),
    Critical = static_cast<int>(spdlog::level::critical),
    Off = static_cast<int>(spdlog::level::off),
  };
  Q_ENUM(LogLevel)

  LogLevel logLevel() const;
  void setLogLevel(LogLevel level);
  spdlog::level::level_enum spdlogLogLevel() const;

  // -- User interface

  double uiScale() const;
  void setUiScale(double value);

  // -- Network

  bool enablePrometheusMetrics() const;
  void setEnablePrometheusMetrics(bool value);

  QString prometheusAddress() const;
  void setPrometheusAddress(const QString &value);

  // -- for plugins that lookup values from a map

  Q_INVOKABLE QVariant value(const QString &key,
                             const QVariant &defaultValue = {}) const;
  Q_INVOKABLE void setValue(const QString &key, const QVariant &value);
  Q_INVOKABLE void sync();

signals:
  void restoreLastSessionChanged();
  void lastSessionPathChanged();

  void logLevelChanged();

  void uiScaleChanged();

  void enablePrometheusMetricsChanged();
  void prometheusAddressChanged();

  void valueChanged(const QString &key);

private:
  template <typename T> T read(const QString &key, const T &def) const {
    return m_settings.value(key, def).template value<T>();
  }

  void write(const QString &key, const QVariant &v);

  static LogLevel logLevelFromString(QStringView levelText);
  static QString logLevelToString(LogLevel level);

  QSettings m_settings;

  bool m_restoreLastSession = true;
  QString m_lastSessionPath;

  LogLevel m_logLevel = LogLevel::Info;

  double m_uiScale = 1.0;

  bool m_enablePrometheusMetrics = true;
  QString m_prometheusAddress;
};

} // namespace pc
