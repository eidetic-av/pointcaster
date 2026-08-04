import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Item {
    id: root

    property string label: ""
    property string description: ""

    default property alias content: controls.data

    readonly property real horizontalPadding: Math.round(10 * Scaling.uiScale)

    readonly property real verticalPadding: Math.round(7 * Scaling.uiScale)

    Layout.fillWidth: true

    implicitHeight: Math.max(Math.round(32 * Scaling.uiScale), labelColumn.implicitHeight + verticalPadding * 2)

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: root.horizontalPadding
        anchors.rightMargin: root.horizontalPadding
        spacing: Math.round(12 * Scaling.uiScale)

        ColumnLayout {
            id: labelColumn

            Layout.fillWidth: true
            Layout.alignment: Qt.AlignVCenter
            spacing: Math.round(2 * Scaling.uiScale)

            Label {
                text: root.label
                visible: text.length > 0
                font: Scaling.uiFont
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }

            Label {
                text: root.description
                visible: text.length > 0
                font: Qt.font({
                    pointSize: Scaling.basePointSize * 0.85 * Scaling.uiScale
                })
                color: ThemeColors.readOnlyText
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }

        RowLayout {
            id: controls

            Layout.alignment: Qt.AlignVCenter
            Layout.preferredHeight: Math.round(26 * Scaling.uiScale)
            spacing: Math.round(8 * Scaling.uiScale)
        }
    }
}
