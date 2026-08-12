#include "log_model.h"

#include <QMetaObject>
#include <QVariantMap>
#include <utility>

namespace pc::ui {

namespace {

QString level_name(spdlog::level::level_enum level) {
  switch (level) {
  case spdlog::level::trace:
    return QStringLiteral("trace");
  case spdlog::level::debug:
    return QStringLiteral("debug");
  case spdlog::level::info:
    return QStringLiteral("info");
  case spdlog::level::warn:
    return QStringLiteral("warning");
  case spdlog::level::err:
    return QStringLiteral("error");
  case spdlog::level::critical:
    return QStringLiteral("critical");
  default:
    return {};
  }
}

} // namespace

LogModel::LogModel(QObject *parent) : QAbstractListModel(parent) {
  _recentEntryTimer.setSingleShot(true);
  connect(&_recentEntryTimer, &QTimer::timeout, this,
          &LogModel::expireRecentEntries);

  pc::set_log_wakeup([this] {
    QMetaObject::invokeMethod(
        this, [this] { takeLogEntries(); }, Qt::QueuedConnection);
  });

  takeLogEntries();
}

LogModel::~LogModel() {
  pc::set_log_wakeup({});
}

int LogModel::rowCount(const QModelIndex &parent) const {
  if (parent.isValid()) return 0;
  return static_cast<int>(_entries.size());
}

QVariant LogModel::data(const QModelIndex &index, int role) const {
  const int row = index.row();
  if (row < 0 || row >= rowCount()) return {};

  const Entry &entry = _entries[row];
  switch (static_cast<Role>(role)) {
  case Role::Level:
    return entry.level;
  case Role::Message:
    return entry.message;
  }
  return {};
}

QHash<int, QByteArray> LogModel::roleNames() const {
  return {
      {static_cast<int>(Role::Level), "level"},
      {static_cast<int>(Role::Message), "message"},
  };
}

QVariantList LogModel::recentEntries() const {
  QVariantList entries;
  entries.reserve(static_cast<qsizetype>(_recentEntries.size()));
  for (const auto &recent : _recentEntries) {
    entries.push_back(
        QVariantMap{{QStringLiteral("level"), recent.entry.level},
                    {QStringLiteral("message"), recent.entry.message}});
  }
  return entries;
}

void LogModel::takeLogEntries() {
  auto pending = pc::take_log_entries();
  if (pending.empty()) return;

  // a burst bigger than the whole scrollback can only fill it once
  if (pending.size() > static_cast<std::size_t>(maxRows))
    pending.erase(pending.begin(), pending.end() - maxRows);

  const auto pending_count = static_cast<int>(pending.size());

  const int overflow =
      static_cast<int>(_entries.size()) + pending_count - maxRows;
  if (overflow > 0) {
    beginRemoveRows({}, 0, overflow - 1);
    _entries.remove(0, overflow);
    endRemoveRows();
  }

  const int first_row = static_cast<int>(_entries.size());
  beginInsertRows({}, first_row, first_row + pending_count - 1);
  for (const auto &log_entry : pending) {
    _entries.push_back(Entry{level_name(log_entry.level),
                             QString::fromStdString(log_entry.message)});
  }
  endInsertRows();

  const auto expires_at =
      std::chrono::steady_clock::now() + recentEntryLifetime;
  for (auto entry = _entries.cend() - pending_count; entry != _entries.cend();
       ++entry) {
    _recentEntries.push_back(RecentEntry{*entry, expires_at});
  }
  while (_recentEntries.size() > static_cast<std::size_t>(maxRecentRows))
    _recentEntries.pop_front();

  emit recentEntriesChanged();
  scheduleRecentEntryExpiry();
}

void LogModel::expireRecentEntries() {
  const auto now = std::chrono::steady_clock::now();
  const auto count_before = _recentEntries.size();

  while (!_recentEntries.empty() && _recentEntries.front().expires_at <= now)
    _recentEntries.pop_front();

  if (_recentEntries.size() != count_before) emit recentEntriesChanged();
  scheduleRecentEntryExpiry();
}

void LogModel::scheduleRecentEntryExpiry() {
  if (_recentEntries.empty()) {
    _recentEntryTimer.stop();
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const auto expires_at = _recentEntries.front().expires_at;
  _recentEntryTimer.start(
      expires_at > now ? std::chrono::duration_cast<std::chrono::milliseconds>(
                             expires_at - now)
                       : std::chrono::milliseconds{0});
}

} // namespace pc::ui
