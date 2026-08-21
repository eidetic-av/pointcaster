import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

Item {
    id: root

    required property var workspace

    readonly property string sessionLabel: (workspace && workspace.selectedSessionAdapter) ? String(workspace.selectedSessionAdapter.label) : ""

    visible: workspace ? workspace.sessionAdapters.length > 1 : false

    implicitHeight: visible ? label.implicitHeight + underline.height : 0

    TextMetrics {
        id: textMetrics
        font: label.font
        text: label.text
    }

    Label {
        id: label
        text: root.sessionLabel

        font: Scaling.uiFont
        color: ThemeColors.text
        opacity: 0.9

        elide: Text.ElideRight
        verticalAlignment: Text.AlignVCenter
        topPadding: Math.round(3 * Scaling.uiScale)

        anchors.left: parent.left
        anchors.right: parent.right
    }

    Rectangle {
        id: underline
        height: 1
        width: Math.min(textMetrics.width, label.width)
        color: ThemeColors.highlight

        anchors {
            left: label.left
            top: label.bottom
            topMargin: Math.round(1 * Scaling.uiScale)
        }
    }
}
