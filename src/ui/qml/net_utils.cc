#include "net_utils.h"

#include <QAbstractSocket>
#include <QHostAddress>
#include <QNetworkInterface>
#include <QString>
#include <qobject.h>

namespace pc::ui::qml {

bool NetUtils::isValidHostAddress(const QString &text) const {
  using namespace Qt::StringLiterals;

  // wildcard addresses with * as hostname just check the port
  if (text.startsWith(u"*:"_s)) {
    bool ok = false;
    const int port = text.mid(2).toInt(&ok);
    return ok && port >= 1024 && port <= 65535;
  }

  // split on last ':'
  const int colonIndex = text.lastIndexOf(u':');
  if (colonIndex <= 0) return false;

  const QString hostPart = text.left(colonIndex).trimmed();
  const QString portPart = text.mid(colonIndex + 1).trimmed();
  if (hostPart.isEmpty() || portPart.isEmpty()) return false;

  bool ok = false;
  const int port = portPart.toInt(&ok);
  if (!ok || port < 1024 || port > 65535) return false; // or 1024..65535 if you want

  QHostAddress addr;
  if (!addr.setAddress(hostPart)) return false;
  if (addr.protocol() != QAbstractSocket::IPv4Protocol) return false;

  return true;
}

QStringList NetUtils::localIpAddresses() const {
  QStringList addresses;
  for (const auto &iface : QNetworkInterface::allInterfaces()) {
    if (!(iface.flags() & QNetworkInterface::IsUp)) continue;
    if (iface.flags() & QNetworkInterface::IsLoopBack) continue;
    for (const auto &entry : iface.addressEntries()) {
      const auto address = entry.ip();
      if (address.protocol() == QAbstractSocket::IPv4Protocol) {
        addresses << address.toString();
      }
    }
  }
  return addresses;
}

} // namespace pc::ui::qml