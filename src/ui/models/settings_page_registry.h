#pragma once

#include <QAbstractListModel>
#include <QHash>
#include <QUrl>
#include <QVector>

namespace pc::ui {

class SettingsPageRegistry final : public QAbstractListModel {
  Q_OBJECT

public:
  enum class Role : int { Key = Qt::UserRole + 1, Title, PageUrl, IsSection };
  Q_ENUM(Role)

  struct PageEntry {
    QString key;
    QString title;
    QUrl pageUrl;
    bool isSection = false;
  };

  static SettingsPageRegistry *instance();

  explicit SettingsPageRegistry(QObject *parent = nullptr);

  int rowCount(const QModelIndex &parent = QModelIndex{}) const override;
  QVariant data(const QModelIndex &index, int role) const override;
  QHash<int, QByteArray> roleNames() const override;

  Q_INVOKABLE int count() const { return rowCount(); }
  Q_INVOKABLE QUrl pageUrlAt(int index) const;

  Q_INVOKABLE bool addPage(const QString &key, const QString &title,
                           const QUrl &pageUrl);
  Q_INVOKABLE bool addSection(const QString &title);
  Q_INVOKABLE bool removePage(const QString &key);
  Q_INVOKABLE void clear();

private:
  int findIndexByKey(const QString &key) const;

  QVector<PageEntry> m_pages;
};

} // namespace pc::ui