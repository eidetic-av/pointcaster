#pragma once

#include <QObject>
#include <recorder/session_recorder.h>

namespace pc::ui {

class RecorderModel : public QObject {
  Q_OBJECT

  Q_PROPERTY(bool recording READ recording NOTIFY recordingChanged)

public:
  explicit RecorderModel(recorder::SessionRecorder *recorder, QObject *parent)
      : QObject(parent), _recorder(recorder) {
    _recorder->set_recording_changed_callback(
        [this](bool isRecording) { recordingChanged(isRecording); });
  }

  int recording() const { return _recorder->recording(); }

public slots:
  void startRecording() { _recorder->start_recording(); }
  void stopRecording() { _recorder->stop_recording(); }

signals:
  void recordingChanged(bool isRecording);

private:
  recorder::SessionRecorder *_recorder;
};

} // namespace pc::ui