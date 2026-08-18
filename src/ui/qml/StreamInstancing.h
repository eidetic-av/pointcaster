#pragma once

#include <QByteArray>
#include <QColor>
#include <QQuick3DInstancing>
#include <QtQmlIntegration/qqmlintegration.h>
#include <ui/models/stream_adapter.h>

namespace pc::ui::qml {

// instances for a model per element in an output stream...
// voxel, aabb, etc. outputs to visualise
class StreamInstancing : public QQuick3DInstancing {
  Q_OBJECT

  QML_NAMED_ELEMENT(StreamInstancing)

  Q_PROPERTY(StreamAdapter *streamAdapter READ streamAdapter WRITE
                 setStreamAdapter NOTIFY streamAdapterChanged)

  // aabbs are coloured by their index so they can be told apart... this is for
  // the streams whose elements have no identity of their own, like voxels
  Q_PROPERTY(QColor color READ color WRITE setColor NOTIFY colorChanged)

  Q_PROPERTY(
      int instanceCount READ instanceCount NOTIFY instanceCountChanged)

  Q_PROPERTY(Kind kind READ kind NOTIFY kindChanged)

public:
  enum Kind { Nothing, Aabbs, Voxels, Bounds };
  Q_ENUM(Kind)

  explicit StreamInstancing(QQuick3DObject *parent = nullptr)
      : QQuick3DInstancing(parent) {}

  StreamAdapter *streamAdapter() const { return _streamAdapter; }
  void setStreamAdapter(StreamAdapter *adapter);

  QColor color() const { return _color; }
  void setColor(const QColor &color);

  int instanceCount() const { return _instanceCount; }

  Kind kind() const { return _kind; }

  // re-reads the stream and redraws it... the scene calls this every time the
  // session it sits in produces a frame
  Q_INVOKABLE void updateInstances();

signals:
  void streamAdapterChanged();
  void colorChanged();
  void instanceCountChanged();
  void kindChanged();

protected:
  QByteArray getInstanceBuffer(int *instanceCount) override;

private:
  StreamAdapter *_streamAdapter = nullptr;
  QColor _color = Qt::white;

  QByteArray _instanceData;
  int _instanceCount = 0;
  Kind _kind = Nothing;
};

} // namespace pc::ui::qml
