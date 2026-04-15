#include "session_recorder_model.h"
#include <QTimer>

namespace pc::ui {

RecorderModel::RecorderModel(recorder::SessionRecorder *recorder,
                             QObject *parent)
    : QObject(parent), _recorder(recorder) {
  _recorder->set_recording_changed_callback(
      [this](bool recording) { isRecordingChanged(recording); });

  // this timer emits UI updates for all vars that change when we
  // are recording
  auto *timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, [this] {
    if (_recorder->is_recording()) {
      emit currentFrameChanged();
      emit currentSecondsChanged();
      emit writerQueueDepthChanged();
      emit droppedFramesChanged();
    }
  });
  timer->start(33); // 30fps updates
}

} // namespace pc::ui