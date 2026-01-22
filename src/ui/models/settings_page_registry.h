#pragma once

#include <QAbstractListModel>
#include <QHash>
#include <QUrl>
#include <QVector>

namespace pc::ui {

class SettingsPageRegistry final : public QAbstractListModel
{
    Q_OBJECT

public:
    enum class Role : int {
        Key = Qt::UserRole + 1,
        Title,
        PageUrl
    };
    Q_ENUM(Role)

    struct PageEntry {
        QString key;
        QString title;
        QUrl pageUrl;
    };

    static SettingsPageRegistry* instance();

    explicit SettingsPageRegistry(QObject* parent = nullptr);

    // QAbstractListModel
    int rowCount(const QModelIndex& parent = QModelIndex{}) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    // Convenience for QML
    Q_INVOKABLE int count() const { return rowCount(); }
    Q_INVOKABLE int indexOfKey(const QString& key) const;
    Q_INVOKABLE QUrl pageUrlAt(int index) const;

    // API for plugins / app
    Q_INVOKABLE bool addPage(const QString& key, const QString& title, const QUrl& pageUrl);
    Q_INVOKABLE bool removePage(const QString& key);
    Q_INVOKABLE void clear();

private:
    int findIndexByKey(const QString& key) const;

    QVector<PageEntry> m_pages;
};

}