#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QtQmlIntegration/qqmlintegration.h>

namespace pc::ui::qml {

// this class is automatically instantiated as a singleton and can be accessed
// from QML using the type name
class NetUtils : public QObject {
  Q_OBJECT
  QML_ELEMENT
  QML_SINGLETON
  QML_NAMED_ELEMENT(NetUtils)
public:
  explicit NetUtils(QObject *parent = nullptr) : QObject(parent) {}
  // check for an ip & port string in a format like '0.0.0.0:8080', '*:4000',
  // '127.0.0.1:9043' etc.
  Q_INVOKABLE bool isValidHostAddress(const QString &text) const;

  // IPv4 addresses of all active, non-loopback network adapters
  Q_INVOKABLE QStringList localIpAddresses() const;
};

} // namespace pc::ui::qml
