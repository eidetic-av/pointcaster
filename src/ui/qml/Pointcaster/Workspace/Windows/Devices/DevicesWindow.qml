import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQml.Models
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

KDDW.DockWidget {
    id: root
    uniqueName: "devicesWindow"
    title: "Devices"

    property var workspace: null

    Item {
        id: devices
        anchors.fill: parent
        property var kddockwidgets_min_size: Qt.size(225, 500)
        clip: true

        // Shared state for the properties pane
        property var currentAdapter: {
            if (!workspace)
                return null;
            const idx = deviceSelectionList.currentIndex;
            if (idx < 0 || idx >= workspace.deviceConfigAdapters.length)
                return null;
            return workspace.deviceConfigAdapters[idx];
        }

        property var otherAdapter: {
            if (!workspace)
                return null;
            const idx = deviceSelectionList.currentIndex + 1;
            if (idx < 0 || idx >= workspace.deviceConfigAdapters.length)
                return null;
            return workspace.deviceConfigAdapters[idx];
        }

        // controller for the currently selected device (index-aligned with adapters)
        property var currentController: (workspace && deviceSelectionList.currentIndex >= 0 && deviceSelectionList.currentIndex < workspace.deviceControllers.length) ? workspace.deviceControllers[deviceSelectionList.currentIndex] : null

        Rectangle {
            anchors.fill: parent
            color: DarkPalette.base
        }

        DevicesToolBar {
            workspace: root.workspace
            id: devicesToolBar
        }

        DeviceSelectionList {
            id: deviceSelectionList
            workspace: root.workspace
            anchors.top: devicesToolBar.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.topMargin: 10
            anchors.bottomMargin: 10
            anchors.leftMargin: 8
            anchors.rightMargin: 8
            height: 160
        }

        Row {
            id: controlRow
            anchors {
                top: deviceSelectionList.bottom
                left: parent.left
                right: parent.right
                topMargin: 10
                leftMargin: 8
                rightMargin: 8
            }
            spacing: 8
            enabled: devices.currentController !== null

            IconButton {
                id: startStopButton

                tooltip: "Start/stop selected device"
                text: {
                    if (!devices.currentController)
                        return "Start";
                    return devices.currentController.status === DevicePluginController.Active ? "Stop" : "Start";
                }

                iconSource: {
                    if (!devices.currentController)
                        return FontAwesome.icon("solid/play");
                    return devices.currentController.status === DevicePluginController.Active ? FontAwesome.icon("solid/stop") : FontAwesome.icon("solid/play");
                }

                enabled: devices.currentController && devices.currentController.status !== DevicePluginController.Loading

                onClicked: {
                    if (!devices.currentController)
                        return;
                    if (devices.currentController.status === DevicePluginController.Active)
                        devices.currentController.stop();
                    else
                        devices.currentController.start();
                }
            }

            IconButton {
                id: restartButton
                tooltip: "Restart selected device"
                text: "Restart"
                iconSource: FontAwesome.icon("solid/rotate-right")

                enabled: devices.currentController && devices.currentController.status === DevicePluginController.Active

                onClicked: {
                    if (devices.currentController)
                        devices.currentController.restart();
                }
            }

            Item {
                width: 1
                Layout.fillWidth: false
            }
        }

        ScrollView {
            id: deviceConfigScrollView
            anchors {
                top: controlRow.bottom
                topMargin: 6
                bottom: parent.bottom
                bottomMargin: 3
                left: parent.left
                right: parent.right
            }
            Component.onCompleted: function () {
                contentItem.boundsBehavior = Flickable.StopAtBounds;
            }

            Column {
                spacing: 6
                width: deviceConfigScrollView.contentItem.width
                height: parent.height

                ConfigurationEditor {
                    model: devices.currentAdapter
                }
                ConfigurationEditor {
                    model: devices.otherAdapter
                }
            }
        }
    }

    Component.onCompleted: function () {
        workspace.triggerDeviceDiscovery();
    }
}
