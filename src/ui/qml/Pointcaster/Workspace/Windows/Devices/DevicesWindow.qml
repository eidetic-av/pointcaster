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
        property var kddockwidgets_min_size: Qt.size(Math.round(225 * Scaling.uiScale), Math.round(500 * Scaling.uiScale))
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

        Row {
            id: controlRow
            anchors {
                top: deviceSelectionList.bottom
                left: parent.left
                right: parent.right
                topMargin: Math.round(10 * Scaling.uiScale)
                leftMargin: Math.round(8 * Scaling.uiScale)
                rightMargin: Math.round(8 * Scaling.uiScale)
            }
            spacing: Math.round(8 * Scaling.uiScale)
            enabled: devices.currentController !== null

            IconButton {
                id: startStopButton

                font: Scaling.uiFont

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

                font: Scaling.uiFont

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
                topMargin: Math.round(6 * Scaling.uiScale)
                bottom: parent.bottom
                bottomMargin: Math.round(3 * Scaling.uiScale)
                left: parent.left
                right: parent.right
            }
            Component.onCompleted: function () {
                contentItem.boundsBehavior = Flickable.StopAtBounds;
            }

            Column {
                spacing: Math.round(6 * Scaling.uiScale)
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
