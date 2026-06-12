#pragma once

#include "enum_adapters.h"
#include <QAbstractListModel>
#include <QString>
#include <QVector>

namespace pc {
class Workspace;
}

namespace pc::ui {

class StreamChannelListModel final : public QAbstractListModel {
  Q_OBJECT

public:
  enum class Role : int { Address = Qt::UserRole + 1, Enabled, Status };
  Q_ENUM(Role)

  struct ChannelEntry {
    QString address;
    bool enabled;
    StreamChannelStatus status = StreamChannelStatus::Disabled;
  };

  explicit StreamChannelListModel(pc::Workspace *workspace,
                                  QObject *parent = nullptr);

  int rowCount(const QModelIndex &parent = QModelIndex{}) const override;
  QVariant data(const QModelIndex &index, int role) const override;
  QHash<int, QByteArray> roleNames() const override;

  // rebuilds the channel list from the workspace's sessions and devices
  void refresh();

  Q_INVOKABLE bool setChannelEnabled(int index, bool enabled);
  Q_INVOKABLE QString channelAddress(int index) const;
  Q_INVOKABLE StreamChannelStatus channelStatus(int index) const;

private slots:
  void pollChannelStatuses();

private:
  pc::Workspace *_workspace;
  QVector<ChannelEntry> _channels;
};

} // namespace pc::ui
