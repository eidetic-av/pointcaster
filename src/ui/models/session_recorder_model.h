#pragma once

#include <QObject>
#include <recorder/session_recorder.h>

namespace pc::ui {

class RecorderModel : public QObject {
  Q_OBJECT

  Q_PROPERTY(bool isRecording READ isRecording NOTIFY isRecordingChanged)
  Q_PROPERTY(bool isFileWriting READ isFileWriting NOTIFY isFileWritingChanged)

  Q_PROPERTY(int currentFrame READ currentFrame NOTIFY currentFrameChanged)
  Q_PROPERTY(float currentSeconds READ currentSeconds NOTIFY currentSecondsChanged)

  Q_PROPERTY(int writerQueueDepth READ writerQueueDepth NOTIFY writerQueueDepthChanged)
  Q_PROPERTY(int droppedFrames READ droppedFrames NOTIFY droppedFramesChanged)

public:
  explicit RecorderModel(recorder::SessionRecorder *recorder, QObject *parent);

  bool isRecording() const { return _recorder->is_recording(); }
  bool isFileWriting() const { return _recorder->is_file_writing(); }

  int currentFrame() const { return _recorder->current_recording_frame(); }
  float currentSeconds() const { return _recorder->current_recording_seconds(); }

  int writerQueueDepth() const { return _recorder->writer_queue_depth();}
  int droppedFrames() const { return _recorder->dropped_frames();}

public slots:
  void startRecording() { _recorder->start_recording(); }
  void stopRecording() { _recorder->stop_recording(); }

signals:
  void isRecordingChanged(bool isRecording);
  void isFileWritingChanged();

  void currentFrameChanged();
  void currentSecondsChanged();

  void writerQueueDepthChanged();
  void droppedFramesChanged();

private:
  recorder::SessionRecorder *_recorder;
};

} // namespace pc::ui