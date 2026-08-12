#pragma once

#include <QByteArray>
#include <QColor>
#include <QQuick3DInstancing>
#include <QtQmlIntegration/qqmlintegration.h>
#include <ui/models/stream_adapter.h>

namespace pc::ui::qml {

// places one instance of a model per element of an operator's output stream:
// one per aabb, or one per occupied voxel, each sized to the element it came
// from. whichever mesh the scene gives the model is what gets drawn
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

  Q_PROPERTY(bool voxelised READ voxelised NOTIFY voxelisedChanged)

public:
  explicit StreamInstancing(QQuick3DObject *parent = nullptr)
      : QQuick3DInstancing(parent) {}

  StreamAdapter *streamAdapter() const { return _streamAdapter; }
  void setStreamAdapter(StreamAdapter *adapter);

  QColor color() const { return _color; }
  void setColor(const QColor &color);

  int instanceCount() const { return _instanceCount; }

  bool voxelised() const { return _voxelised; }

  // re-reads the stream and redraws it... the scene calls this every time the
  // session it sits in produces a frame
  Q_INVOKABLE void updateInstances();

signals:
  void streamAdapterChanged();
  void colorChanged();
  void instanceCountChanged();
  void voxelisedChanged();

protected:
  QByteArray getInstanceBuffer(int *instanceCount) override;

private:
  StreamAdapter *_streamAdapter = nullptr;
  QColor _color = Qt::white;

  QByteArray _instanceData;
  int _instanceCount = 0;
  bool _voxelised = false;
};

} // namespace pc::ui::qml
