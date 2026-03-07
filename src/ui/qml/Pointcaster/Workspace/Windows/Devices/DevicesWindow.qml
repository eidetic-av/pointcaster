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
    Component.onCompleted: function() {
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

        Row {
            id: controlRow

            spacing: Math.round(8 * Scaling.uiScale)
            enabled: devices.currentAdapter !== null

            anchors {
                top: deviceSelectionList.bottom
                left: parent.left
                right: parent.right
                topMargin: Math.round(10 * Scaling.uiScale)
                leftMargin: Math.round(8 * Scaling.uiScale)
                rightMargin: Math.round(8 * Scaling.uiScale)
            }

            IconButton {
                id: startStopButton

                font: Scaling.uiFont
                tooltip: "Start/stop selected device"
                text: {
                    if (!devices.currentAdapter)
                        return "Start";

                    return devices.currentAdapter.status === UiEnums.WorkspaceDeviceStatus.Active ? "Stop" : "Start";
                }
                iconSource: {
                    if (!devices.currentAdapter)
                        return FontAwesome.icon("solid/play");

                    return devices.currentAdapter.status === UiEnums.WorkspaceDeviceStatus.Active ? FontAwesome.icon("solid/stop") : FontAwesome.icon("solid/play");
                }
                enabled: devices.currentAdapter && devices.currentAdapter.status && devices.currentAdapter.status !== UiEnums.WorkspaceDeviceStatus.Loading
                onClicked: {
                    if (!devices.currentAdapter)
                        return ;

                    if (devices.currentAdapter.status === UiEnums.WorkspaceDeviceStatus.Active)
                        devices.currentAdapter.stop();
                    else
                        devices.currentAdapter.start();
                }
            }

            IconButton {
                id: restartButton

                font: Scaling.uiFont
                tooltip: "Restart selected device"
                text: "Restart"
                iconSource: FontAwesome.icon("solid/rotate-right")
                enabled: devices.currentAdapter && devices.currentAdapter.status && devices.currentAdapter.status === UiEnums.WorkspaceDeviceStatus.Active
                onClicked: {
                    if (devices.currentAdapter)
                        devices.currentAdapter.restart();

                }
            }

            Item {
                width: 1
                Layout.fillWidth: false
            }

        }

        ScrollView {
            id: deviceConfigScrollView

            Component.onCompleted: function() {
                contentItem.boundsBehavior = Flickable.StopAtBounds;
            }

            anchors {
                top: controlRow.bottom
                topMargin: Math.round(6 * Scaling.uiScale)
                bottom: parent.bottom
                bottomMargin: Math.round(3 * Scaling.uiScale)
                left: parent.left
                right: parent.right
            }

            Column {
                spacing: Math.round(6 * Scaling.uiScale)
                width: deviceConfigScrollView.contentItem.width
                height: parent.height

                ConfigurationEditor {
                    model: devices.currentAdapter
                    flattenFields: false
                }

            }

        }

    }

}
