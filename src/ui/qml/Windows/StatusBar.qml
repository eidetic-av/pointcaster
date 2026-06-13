import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Rectangle {
    id: root

    implicitHeight: Math.round(22 * Scaling.uiScale)
    color: ThemeColors.base

    function fileNameFromUrl(url) {
        const path = url.toString();
        if (path.length === 0)
            return "Untitled";
        return decodeURIComponent(path.substring(path.lastIndexOf("/") + 1));
    }

    Timer {
        interval: 5000
        running: true
        repeat: true
        triggeredOnStart: true
        onTriggered: ipAddressesLabel.text = NetUtils.localIpAddresses().join(", ")
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 8 * Scaling.uiScale
        anchors.rightMargin: 8 * Scaling.uiScale
        spacing: 8 * Scaling.uiScale

        Label {
            text: root.fileNameFromUrl(workspaceModel.saveFileUrl)
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
    }
}
