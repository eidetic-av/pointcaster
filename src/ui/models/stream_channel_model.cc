#include "stream_channel_model.h"

namespace pc::ui {

StreamChannelListModel::StreamChannelListModel(QObject *parent)
    : QAbstractListModel(parent) {
  _channels.push_back({QStringLiteral("session_1"), true});
}

int StreamChannelListModel::rowCount(const QModelIndex &parent) const {
  if (parent.isValid()) return 0;
  return static_cast<int>(_channels.size());
}

QVariant StreamChannelListModel::data(const QModelIndex &index,
                                      int role) const {
  if (!index.isValid()) return {};
  const int row = index.row();
  if (row < 0 || row >= rowCount()) return {};

  const ChannelEntry &e = _channels[row];
  switch (static_cast<Role>(role)) {
  case Role::Address:
    return e.address;
  case Role::Enabled:
    return e.enabled;
  case Role::Status:
    return static_cast<int>(channelStatus(row));
  }
  return {};
}

QHash<int, QByteArray> StreamChannelListModel::roleNames() const {
  return {
      {static_cast<int>(Role::Address), "address"},
      {static_cast<int>(Role::Enabled), "enabled"},
      {static_cast<int>(Role::Status), "status"},
  };
}

bool StreamChannelListModel::setChannelEnabled(int index, bool enabled) {
  if (index < 0 || index >= rowCount()) return false;
  if (_channels[index].enabled == enabled) return false;
  _channels[index].enabled = enabled;
  const auto modelIndex = this->index(index);
  emit dataChanged(modelIndex, modelIndex,
                    {static_cast<int>(Role::Enabled),
                     static_cast<int>(Role::Status)});
  return true;
}

QString StreamChannelListModel::channelAddress(int index) const {
  if (index < 0 || index >= rowCount()) return {};
  return _channels[index].address;
}

StreamChannelStatus StreamChannelListModel::channelStatus(int index) const {
  if (index < 0 || index >= rowCount()) return StreamChannelStatus::Disabled;
  const ChannelEntry &e = _channels[index];
  if (!e.enabled) return StreamChannelStatus::Disabled;
  return e.hasListeners ? StreamChannelStatus::Connected
                         : StreamChannelStatus::Live;
}

} // namespace pc::ui
