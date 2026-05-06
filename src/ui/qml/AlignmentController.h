#pragma once

#include <pointcaster/core_types.h>

#include <QMatrix4x4>
#include <QObject>
#include <QQuaternion>
#include <QVariantList>
#include <QVector3D>
#include <QtQmlIntegration/qqmlintegration.h>

namespace pc::ui::qml {

class AlignmentController : public QObject {
  Q_OBJECT
  QML_NAMED_ELEMENT(AlignmentController)

  Q_PROPERTY(bool hasResult READ hasResult NOTIFY resultChanged)
  Q_PROPERTY(QMatrix4x4 transform READ transform NOTIFY resultChanged)
  Q_PROPERTY(QVector3D resultPosition READ resultPosition NOTIFY resultChanged)
  Q_PROPERTY(
      QQuaternion resultRotation READ resultRotation NOTIFY resultChanged)

public:
  explicit AlignmentController(QObject *parent = nullptr);

  bool hasResult() const { return _hasResult; }
  QMatrix4x4 transform() const { return _transform; }
  QVector3D resultPosition() const { return _position; }
  QQuaternion resultRotation() const { return _rotation; }

  // pairs: [{primary: vector3d, secondary: vector3d}, ...]
  // Computes transform that maps secondary to primary.
  Q_INVOKABLE void computeFromPairs(const QVariantList &pairs);

  Q_INVOKABLE void reset();

signals:
  void resultChanged();

private:
  bool _hasResult = false;
  QMatrix4x4 _transform;
  QVector3D _position;
  QQuaternion _rotation;

  void decomposeTransform(const pc::float4x4 &matrix);
};

} // namespace pc::ui::qml