import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Item {
    id: root

    property string title: ""

    default property alias content: body.data

    Rectangle {
        anchors.fill: parent
        color: ThemeColors.window
    }

    PaddedScrollView {
        id: scroll

        anchors.fill: parent
        clip: true
        padding: Math.round(16 * Scaling.uiScale)
        contentWidth: availableWidth

        ColumnLayout {
            id: body

            width: scroll.availableWidth
            spacing: Math.round(16 * Scaling.uiScale)

            Label {
                text: root.title
                font: Qt.font({
                    pointSize: Scaling.basePointSize * 1.15 * Scaling.uiScale,
                    weight: Font.DemiBold
                })
                elide: Text.ElideRight
                Layout.fillWidth: true
            }

            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 1
                color: ThemeColors.middark
            }
        }
    }
}
