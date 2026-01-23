#pragma once

#include <QObject>
#include <QSettings>
#include <QVariant>

namespace pc {

class AppSettings final : public QObject {
  Q_OBJECT

  Q_PROPERTY(bool restoreLastSession READ restoreLastSession WRITE
                 setRestoreLastSession NOTIFY restoreLastSessionChanged)
  Q_PROPERTY(QString lastSessionPath READ lastSessionPath WRITE
                 setLastSessionPath NOTIFY lastSessionPathChanged)

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

  void uiScaleChanged();

  void enablePrometheusMetricsChanged();
  void prometheusAddressChanged();

  void valueChanged(const QString &key);

private:
  template <typename T> T read(const QString &key, const T &def) const {
    return m_settings.value(key, def).template value<T>();
  }

  void write(const QString &key, const QVariant &v);

  QSettings m_settings;

  bool m_restoreLastSession = true;
  QString m_lastSessionPath;

  double m_uiScale = 1.0;

  bool m_enablePrometheusMetrics = true;
  QString m_prometheusAddress;
};

} // namespace pc