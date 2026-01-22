#pragma once

#include <QObject>
#include <QSettings>
#include <QVariant>

namespace pc {

class AppSettings final : public QObject
{
    Q_OBJECT
    Q_PROPERTY(double uiScale READ uiScale WRITE setUiScale NOTIFY uiScaleChanged)
    Q_PROPERTY(bool restoreLastSession READ restoreLastSession WRITE setrestoreLastSession NOTIFY restoreLastSessionChanged)

public:
    static AppSettings* instance();

    explicit AppSettings(QObject* parent = nullptr);

    double uiScale() const;
    void setUiScale(double value);

    bool restoreLastSession() const;
    void setrestoreLastSession(bool value);

    Q_INVOKABLE QVariant value(const QString& key, const QVariant& defaultValue = {}) const;
    Q_INVOKABLE void setValue(const QString& key, const QVariant& value);
    Q_INVOKABLE void sync();

signals:
    void uiScaleChanged();
    void restoreLastSessionChanged();

    // For generic key observers (optional)
    void valueChanged(const QString& key);

private:
    template <typename T>
    T read(const QString& key, const T& def) const {
        return m_settings.value(key, def).template value<T>();
    }

    void write(const QString& key, const QVariant& v);

    QSettings m_settings;

    double  m_uiScale = 1.0;
    bool    m_restoreLastSession = true;
};

}