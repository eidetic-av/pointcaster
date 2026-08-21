import QtQuick

import Pointcaster 1.0

Item {
    id: root

    property real dropLineX: 0
    property color accent: ThemeColors.highlight
    property bool showBackground: true

    readonly property real dropLineWidth: Math.round(2 * Scaling.uiScale)

    Rectangle {
        anchors.fill: parent
        visible: root.showBackground
        color: Qt.rgba(root.accent.r, root.accent.g, root.accent.b, 0.12)
        border.color: Qt.rgba(root.accent.r, root.accent.g, root.accent.b, 0.55)
        border.width: 1
    }

    Rectangle {
        color: root.accent
        x: root.dropLineX - root.dropLineWidth / 2
        y: 0
        width: root.dropLineWidth
        height: root.height
    }
}
