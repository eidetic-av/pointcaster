import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Pointcaster 1.0

Item {
    id: root

    required property var adapter

    readonly property var seq: adapter && adapter.hasSequence ? adapter.sequenceAdapter : null

    // ── Frame state from DeviceAdapter (updated per-frame via tick) ──
    readonly property int totalFrames: adapter ? adapter.frameCount : 1
    readonly property int currentFrame: adapter ? adapter.currentFrame : 0
    readonly property bool isPlaying: adapter ? adapter.isPlaying : false

    // ── Config values from seq Q_PROPERTYs (reactive via NOTIFY signals) ──
    readonly property bool isLooping: seq ? seq.looping : false
    readonly property int startFrame: seq ? seq.start_frame : 0
    readonly property int endFrame: {
        if (!seq)
            return totalFrames - 1;
        var e = seq.end_frame;
        return e < 0 ? totalFrames - 1 : e;
    }

    visible: seq !== null
    implicitHeight: visible ? layout.implicitHeight : 0

    ColumnLayout {
        id: layout
        anchors.fill: parent
        spacing: Math.round(4 * Scaling.uiScale)

        // ── Transport buttons ──
        RowLayout {
            spacing: Math.round(6 * Scaling.uiScale)
            Layout.fillWidth: true

            IconButton {
                tooltip: "Stop"
                iconSource: FontAwesome.icon("solid/stop")
                iconSize: Math.round(11 * Scaling.uiScale)
                onClicked: {
                    seq.set("playing", false);
                    seq.set("current_frame", root.startFrame);
                }
            }

            IconButton {
                tooltip: "Play"
                iconSource: FontAwesome.icon("solid/play")
                iconSize: Math.round(11 * Scaling.uiScale)
                enabled: !root.isPlaying
                opacity: enabled ? 1.0 : 0.4
                onClicked: seq.set("playing", true)
            }

            IconButton {
                tooltip: "Pause"
                iconSource: FontAwesome.icon("solid/pause")
                iconSize: Math.round(11 * Scaling.uiScale)
                enabled: root.isPlaying
                opacity: enabled ? 1.0 : 0.4
                onClicked: seq.set("playing", false)
            }

            IconButton {
                tooltip: root.isLooping ? "Looping" : "Play once"
                iconSource: FontAwesome.icon("solid/repeat")
                iconSize: Math.round(11 * Scaling.uiScale)
                opacity: root.isLooping ? 1.0 : 0.4
                onClicked: seq.set("looping", !root.isLooping)
            }

            Item {
                Layout.fillWidth: true
            }

            Text {
                text: root.currentFrame + " / " + (root.totalFrames - 1)
                font: Scaling.uiSmallFont
                color: ThemeColors.text
                opacity: 0.7
            }
        }

        // ── Timeline track ──
        Item {
            id: track
            Layout.fillWidth: true
            Layout.preferredHeight: Math.round(28 * Scaling.uiScale)

            readonly property real handleSize: Math.round(6 * Scaling.uiScale)
            readonly property real usableWidth: width - handleSize

            function frameToX(frame) {
                if (root.totalFrames <= 1)
                    return 0;
                return (frame / (root.totalFrames - 1)) * usableWidth;
            }
            function xToFrame(x) {
                if (usableWidth <= 0)
                    return 0;
                return Math.round(Math.max(0, Math.min(1, x / usableWidth)) * (root.totalFrames - 1));
            }

            // ── Rail ──
            Rectangle {
                id: rail
                anchors.verticalCenter: parent.verticalCenter
                width: parent.width
                height: Math.round(4 * Scaling.uiScale)
                radius: height / 2
                color: ThemeColors.almostdark
            }

            // ── Loop region ──
            Rectangle {
                anchors.verticalCenter: rail.verticalCenter
                height: rail.height
                radius: rail.radius
                color: ThemeColors.highlight
                opacity: 0.25
                x: track.frameToX(root.startFrame) + track.handleSize / 2
                width: Math.max(0, track.frameToX(root.endFrame) - track.frameToX(root.startFrame))
            }

            // ── In handle ──
            Rectangle {
                id: inHandle
                width: track.handleSize
                height: parent.height
                radius: 2
                color: ThemeColors.highlight
                opacity: inArea.containsMouse || inArea.pressed ? 1.0 : 0.6

                // driven by binding when not dragging, by drag when dragging
                property bool dragging: false
                x: dragging ? x : track.frameToX(root.startFrame)

                MouseArea {
                    id: inArea
                    anchors.fill: parent
                    anchors.margins: Math.round(-4 * Scaling.uiScale)
                    hoverEnabled: true
                    cursorShape: Qt.SizeHorCursor

                    property real dragStartX
                    property real dragStartMouseX

                    onPressed: function (mouse) {
                        inHandle.dragging = true;
                        dragStartX = inHandle.x;
                        dragStartMouseX = mouse.x;
                    }
                    onPositionChanged: function (mouse) {
                        if (!pressed)
                            return;
                        var newX = Math.max(0, Math.min(outHandle.x - track.handleSize, dragStartX + (mouse.x - dragStartMouseX)));
                        inHandle.x = newX;
                        seq.set("start_frame", track.xToFrame(newX));
                    }
                    onReleased: inHandle.dragging = false
                    onCanceled: inHandle.dragging = false
                }

                InfoToolTip {
                    textValue: "Loop in: " + root.startFrame
                }
            }

            // ── Out handle ──
            Rectangle {
                id: outHandle
                width: track.handleSize
                height: parent.height
                radius: 2
                color: ThemeColors.highlight
                opacity: outArea.containsMouse || outArea.pressed ? 1.0 : 0.6

                property bool dragging: false
                x: dragging ? x : track.frameToX(root.endFrame)

                MouseArea {
                    id: outArea
                    anchors.fill: parent
                    anchors.margins: Math.round(-4 * Scaling.uiScale)
                    hoverEnabled: true
                    cursorShape: Qt.SizeHorCursor

                    property real dragStartX
                    property real dragStartMouseX

                    onPressed: function (mouse) {
                        outHandle.dragging = true;
                        dragStartX = outHandle.x;
                        dragStartMouseX = mouse.x;
                    }
                    onPositionChanged: function (mouse) {
                        if (!pressed)
                            return;
                        var newX = Math.max(inHandle.x + track.handleSize, Math.min(track.usableWidth, dragStartX + (mouse.x - dragStartMouseX)));
                        outHandle.x = newX;
                        seq.set("end_frame", track.xToFrame(newX));
                    }
                    onReleased: outHandle.dragging = false
                    onCanceled: outHandle.dragging = false
                }

                InfoToolTip {
                    textValue: "Loop out: " + root.endFrame
                }
            }

            // ── Playhead ──
            Rectangle {
                id: playhead
                width: Math.round(3 * Scaling.uiScale)
                height: parent.height
                radius: 1
                color: ThemeColors.text

                readonly property real offset: (track.handleSize - width) / 2

                x: scrubArea.scrubbing ? track.frameToX(scrubArea.scrubFrame) + offset : track.frameToX(root.currentFrame) + offset

                // Behavior on x {
                //     enabled: !scrubArea.scrubbing
                //     NumberAnimation {
                //         duration: 32
                //     }
                // }
            }

            // ── Scrub area ──
            MouseArea {
                id: scrubArea
                anchors.fill: parent
                z: -1
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor

                property bool scrubbing: false
                property int scrubFrame: 0
                property bool wasPlaying: false

                function clampedFrame(mouseX) {
                    var f = track.xToFrame(mouseX);
                    return Math.max(root.startFrame, Math.min(root.endFrame, f));
                }

                onPressed: function (mouse) {
                    wasPlaying = root.isPlaying;
                    scrubbing = true;
                    if (wasPlaying)
                        seq.set("playing", false);
                    scrubFrame = clampedFrame(mouse.x);
                    seq.set("current_frame", scrubFrame);
                }
                onPositionChanged: function (mouse) {
                    if (!scrubbing)
                        return;
                    scrubFrame = clampedFrame(mouse.x);
                    seq.set("current_frame", scrubFrame);
                }
                onReleased: {
                    scrubbing = false;
                    if (wasPlaying)
                        seq.set("playing", true);
                }
                onCanceled: scrubbing = false
            }
        }
    }
}
