#pragma once

#include "enum_adapters.h"
#include <QAbstractListModel>
#include <QString>
#include <QVector>

namespace pc::ui {

class StreamChannelListModel final : public QAbstractListModel {
  Q_OBJECT

public:
  enum class Role : int { Address = Qt::UserRole + 1, Enabled, Status };
  Q_ENUM(Role)

  struct ChannelEntry {
    QString address;
    bool enabled;
    // TODO set from real XPUB listener state once that lands
    bool hasListeners = false;
  };

  explicit StreamChannelListModel(QObject *parent = nullptr);

  int rowCount(const QModelIndex &parent = QModelIndex{}) const override;
  QVariant data(const QModelIndex &index, int role) const override;
  QHash<int, QByteArray> roleNames() const override;

  // TODO wire this into the real per-channel
  // enable state in workspace config
  Q_INVOKABLE bool setChannelEnabled(int index, bool enabled);
  Q_INVOKABLE QString channelAddress(int index) const;
  Q_INVOKABLE StreamChannelStatus channelStatus(int index) const;

private:
  QVector<ChannelEntry> _channels;
};

} // namespace pc::ui
