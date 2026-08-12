pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

Item {
    id: root

    // pc::ui::LogModel
    required property var logModel

    property bool collapsed: true
    property int panelWidth: Math.round(500 * Scaling.uiScale)

    // once dragged out to the full view width, stay there as the view resizes
    property bool fullWidth: false

    readonly property int panelHeight: Math.round(200 * Scaling.uiScale)
    readonly property int panelWidthMin: Math.round(200 * Scaling.uiScale)
    readonly property int panelWidthMax: root.width - collapser.container.spacing

    onWidthChanged: if (root.fullWidth)
        root.panelWidth = root.panelWidthMax

    SessionControlCollapser {
        id: collapser

        direction: SessionControlCollapser.CollapseDown

        anchors.right: parent.right
        anchors.bottom: parent.bottom

        collapsed: root.collapsed
        onCollapsedChanged: root.collapsed = collapsed

        panelRadius: root.fullWidth ? 0 : Math.round(7 * Scaling.uiScale)

        contentItem: ListView {
            id: scrollback

            width: root.panelWidth
            height: root.panelHeight
            clip: true

            model: root.logModel
            spacing: 0
            boundsBehavior: Flickable.StopAtBounds

            // collapsed so don't take the scroll wheel
            interactive: !root.collapsed

            ScrollBar.vertical: PaddedScrollBar {
                id: scrollbackBar
                barEnabled: !root.collapsed
            }

            property bool followTail: true

            delegate: LogEntry {
                required property var model
                width: scrollback.width - scrollbackBar.width
                level: model.level
                message: model.message
            }

            onCountChanged: if (followTail)
                Qt.callLater(positionViewAtEnd)

            onHeightChanged: if (followTail)
                Qt.callLater(positionViewAtEnd)

            Connections {
                target: scrollback.model
                function onRowsAboutToBeInserted() {
                    scrollback.followTail = scrollback.atYEnd || scrollback.contentHeight <= scrollback.height;
                }
            }
        }
    }

    // the newest entries, floating over the view while the panel is shut
    Item {
        id: overlay

        visible: collapser.container.height === 0

        width: root.panelWidth
        height: overlayColumn.height

        x: parent.width - width
        y: collapser.y - height

        Rectangle {
            anchors.fill: parent
            color: ThemeColors.dark
            opacity: 0.35
        }

        Column {
            id: overlayColumn

            width: parent.width
            spacing: 0

            Repeater {
                model: root.logModel ? root.logModel.recentEntries : []

                LogEntry {
                    required property var modelData

                    width: overlayColumn.width
                    level: modelData.level
                    message: modelData.message
                }
            }
        }
    }

    // drag the panel's left edge to resize it
    Item {
        id: resizeBar

        width: Math.round(10 * Scaling.uiScale)
        height: collapser.container.height

        x: collapser.x - Math.round(width / 2)
        y: collapser.y + collapser.container.y

        MouseArea {
            anchors.fill: parent
            z: 50

            hoverEnabled: true
            acceptedButtons: Qt.LeftButton
            cursorShape: Qt.SplitHCursor
            preventStealing: true

            property bool dragging: false
            onPressed: dragging = true
            onReleased: dragging = false
            onCanceled: dragging = false

            onPositionChanged: {
                if (!dragging)
                    return;
                let newWidth = Math.round(root.panelWidth - mouseX);
                newWidth = Math.max(newWidth, root.panelWidthMin);
                newWidth = Math.min(newWidth, root.panelWidthMax);
                root.panelWidth = newWidth;
                root.fullWidth = newWidth >= root.panelWidthMax - resizeBar.width;
            }
        }
    }
}
