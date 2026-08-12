#pragma once

#include <QAbstractListModel>
#include <QList>
#include <QString>
#include <QTimer>
#include <QVariantList>
#include <chrono>
#include <core/logger/logger.h>
#include <deque>

namespace pc::ui {

class LogModel final : public QAbstractListModel {
  Q_OBJECT

  Q_PROPERTY(
      QVariantList recentEntries READ recentEntries NOTIFY recentEntriesChanged)

public:
  enum class Role : int { Level = Qt::UserRole + 1, Message };
  Q_ENUM(Role)

  explicit LogModel(QObject *parent = nullptr);
  ~LogModel() override;

  int rowCount(const QModelIndex &parent = QModelIndex{}) const override;
  QVariant data(const QModelIndex &index, int role) const override;
  QHash<int, QByteArray> roleNames() const override;

  QVariantList recentEntries() const;

signals:
  void recentEntriesChanged();

private:
  // scrollback kept for the expanded scrolling panel
  static constexpr int maxRows = 1000;
  static constexpr int maxRecentRows = 6;
  static constexpr std::chrono::milliseconds recentEntryLifetime{10000};

  struct Entry {
    QString level;
    QString message;
  };

  struct RecentEntry {
    Entry entry;
    std::chrono::steady_clock::time_point expires_at;
  };

  void takeLogEntries();
  void expireRecentEntries();
  void scheduleRecentEntryExpiry();

  QList<Entry> _entries;

  std::deque<RecentEntry> _recentEntries;
  QTimer _recentEntryTimer;
};

} // namespace pc::ui
