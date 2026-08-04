import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

ColumnLayout {
    id: root

    property string title: ""
    property string description: ""

    default property alias content: rows.data

    Layout.fillWidth: true
    spacing: Math.round(6 * Scaling.uiScale)

    Label {
        text: root.title
        visible: text.length > 0
        font: Qt.font({
            pointSize: Scaling.pointSize,
            weight: Font.DemiBold
        })
        elide: Text.ElideRight
        Layout.fillWidth: true
    }

    Label {
        text: root.description
        visible: text.length > 0
        font: Scaling.uiFont
        color: ThemeColors.readOnlyText
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    Rectangle {
        Layout.fillWidth: true

        implicitHeight: rows.implicitHeight
        color: ThemeColors.withAlpha(ThemeColors.mid, 0.18)
        radius: Math.round(4 * Scaling.uiScale)

        ColumnLayout {
            id: rows

            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            spacing: 0
        }
    }
}
