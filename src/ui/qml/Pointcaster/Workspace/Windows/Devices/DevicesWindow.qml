import Pointcaster 1.0
import Pointcaster.Workspace 1.0
import QtQml.Models
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

KDDW.DockWidget {
    id: root

    property var workspace: null

    uniqueName: "devicesWindow"
    title: "Devices"
    Component.onCompleted: function () {
        if (workspace)
            workspace.triggerDeviceDiscovery();
    }

    Item {
        id: devices

        property var kddockwidgets_min_size: Qt.size(Math.round(225 * Scaling.uiScale), Math.round(500 * Scaling.uiScale))
        property var currentAdapter: {
            if (!workspace)
                return null;

            const idx = deviceSelectionList.currentIndex;
            if (idx < 0 || idx >= workspace.deviceAdapters.length)
                return null;

            return workspace.deviceAdapters[idx];
        }

        anchors.fill: parent
        clip: true

        Rectangle {
            anchors.fill: parent
            color: ThemeColors.base
        }

        DevicesToolBar {
            id: devicesToolBar
            workspace: root.workspace
        }

        DeviceSelectionList {
            id: deviceSelectionList

            workspace: root.workspace
            anchors.top: devicesToolBar.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.topMargin: Math.round(10 * Scaling.uiScale)
            anchors.bottomMargin: Math.round(10 * Scaling.uiScale)
            anchors.leftMargin: Math.round(8 * Scaling.uiScale)
            anchors.rightMargin: Math.round(8 * Scaling.uiScale)
            height: Math.round(160 * Scaling.uiScale)
        }

        // Rectangle {
        //     color: "green"
        //     width: 200
        //     anchors {
        //         top: deviceSelectionList.bottom
        //         bottom: devices.bottom
        //     }
        // }

        DeviceConfigEditor {
            id: configEditor
            adapter: devices.currentAdapter
            anchors {
                top: deviceSelectionList.bottom
                bottom: devices.bottom
                left: devices.left
                right: devices.right
                topMargin: Math.round(10 * Scaling.uiScale)
                bottomMargin: Math.round(8 * Scaling.uiScale)
                leftMargin: Math.round(8 * Scaling.uiScale)
                rightMargin: Math.round(8 * Scaling.uiScale)
            }
        }
    }
}
