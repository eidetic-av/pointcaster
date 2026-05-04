import Pointcaster.Workspace 1.0

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

KDDW.DockWidget {
    id: root
    uniqueName: "recordingWindow"
    title: "Recording"

    required property var workspace
    required property var recorder

    Item {
        anchors.fill: parent

        Rectangle {
            id: background
            anchors.fill: parent
            color: ThemeColors.base
        }

        RowLayout {
            property var minWidth: Math.round(400 * Scaling.uiScale)
            property var minHeight: Math.round(150 * Scaling.uiScale)
            property var kddockwidgets_min_size: Qt.size(minWidth, minHeight)

            anchors.fill: parent

            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true

                Column {
                    Layout.alignment: Qt.AlignVCenter
                    width: Math.round(100 * Scaling.uiScale)

                    Label {
                        text: "Queue depth: " + recorder.writerQueueDepth
                    }

                    Label {
                        text: "Dropped frames: " + recorder.droppedFrames
                    }
                }
            }

            Column {
                property var recordColumnWidth: Math.round(100 * Scaling.uiScale)

                Layout.minimumWidth: recordColumnWidth
                Layout.preferredWidth: recordColumnWidth
                Layout.maximumWidth: recordColumnWidth

                Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter

                spacing: Math.round(6 * Scaling.uiScale)

                Rectangle {
                    id: recordingButton
                    width: Math.round(50 * Scaling.uiScale)
                    height: Math.round(50 * Scaling.uiScale)
                    radius: Math.round(width * 0.5)
                    color: ThemeColors.dark
                    border.color: ThemeColors.midlight
                    border.width: 1

                    property var innerShapePadding: Math.round(8 * Scaling.uiScale)
                    property var recordingCircleSize: width - innerShapePadding
                    property var recordingCircleRadius: Math.round(recordingCircleSize * 0.5)
                    property var stopButtonSize: Math.round(recordingCircleSize * 0.75)
                    property var stopButtonRadius: Math.round(4 * Scaling.uiScale)

                    Rectangle {
                        id: recordingStatusShape
                        width: recordingButton.recordingCircleSize
                        height: recordingButton.recordingCircleSize
                        radius: recordingButton.recordingCircleRadius
                        anchors.centerIn: parent
                        color: ThemeColors.red

                        states: [
                            State {
                                name: "idle"
                                when: !recorder.isRecording
                                PropertyChanges {
                                    target: recordingStatusShape
                                    radius: recordingButton.recordingCircleRadius
                                    width: recordingButton.recordingCircleSize
                                    height: recordingButton.recordingCircleSize
                                    color: ThemeColors.red
                                    rotation: 45
                                }
                            },
                            State {
                                name: "recording"
                                when: recorder.isRecording
                                PropertyChanges {
                                    target: recordingStatusShape
                                    radius: recordingButton.stopButtonRadius
                                    width: recordingButton.stopButtonSize
                                    height: recordingButton.stopButtonSize
                                    color: ThemeColors.light
                                    rotation: 0
                                }
                            }
                        ]

                        transitions: Transition {
                            NumberAnimation {
                                properties: "radius,width,height,rotation"
                                duration: 100
                                easing.type: Easing.InOutQuad
                            }
                            ColorAnimation {
                                property: "color"
                                duration: 100
                                easing.type: Easing.InOutQuad
                            }
                        }
                    }

                    MouseArea {
                        id: recordingButtonMouseArea
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: {
                            if (!recorder.isRecording)
                                recorder.startRecording();
                            else if (recorder.isRecording)
                                recorder.stopRecording();
                        }
                    }

                    InfoToolTip {
                        visible: recordingButtonMouseArea.hovered
                        textValue: (!recorder.isRecording ? "Start" : "Stop") + " recording"
                    }
                }

                Label {
                    text: "Frame " + recorder.currentFrame
                }

                Label {
                    text: recorder.currentSeconds.toFixed(2) + " seconds"
                }
            }
        }
    }
}
