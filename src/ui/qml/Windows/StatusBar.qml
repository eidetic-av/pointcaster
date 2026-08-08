import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Rectangle {
    id: root

    implicitHeight: Math.round(22 * Scaling.uiScale)
    color: ThemeColors.base

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 8 * Scaling.uiScale
        anchors.rightMargin: 8 * Scaling.uiScale
        spacing: 8 * Scaling.uiScale

        Label {
            text: workspaceModel.saveFileUrl
            font: Scaling.uiFont
            color: ThemeColors.midlight
        }

        Item {
            Layout.fillWidth: true
        }

        Label {
            id: ipAddressesLabel
            font: Scaling.uiFont
            color: ThemeColors.midlight
        }

        Component.onCompleted: {
            ipAddressesLabel.text = NetUtils.localIpAddresses().join(", ");
        }
    }
}
