import Pointcaster 1.0
import Pointcaster.Workspace 1.0
import QtQml.Models
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

KDDW.DockWidget {
    id: root
    uniqueName: "devicesWindow"
    title: "Devices"

    property var workspace: null

    function init() {
        if (workspace) {
            workspace.triggerDeviceDiscovery();
            deviceSelectionList.setSelectedIndex(workspace.selectedDeviceIndex);
        }
    }

    Component.onCompleted: init()

    Connections {
        target: workspace
        function onNewWorkspaceLoaded() {
            init();
        }
    }

    // when a new device is added, make sure its selected
    Connections {
        target: workspace
        function onDeviceAdded() {
            deviceSelectionList.setSelectedIndex(workspace.deviceAdapters.length - 1);
        }
    }

    Item {
        anchors.fill: parent

        Rectangle {
            id: background
            anchors.fill: parent
            color: ThemeColors.base
        }

        ColumnLayout {
            id: content

            property var kddockwidgets_min_size: Qt.size(Math.round(225 * Scaling.uiScale), Math.round(500 * Scaling.uiScale))

            anchors {
                fill: parent
                topMargin: Math.round(10 * Scaling.uiScale)
                bottomMargin: Math.round(10 * Scaling.uiScale)
                leftMargin: Math.round(8 * Scaling.uiScale)
                rightMargin: Math.round(8 * Scaling.uiScale)
            }

            spacing: Math.round(10 * Scaling.uiScale)

            DevicesToolBar {
                id: devicesToolBar
                workspace: root.workspace
                Layout.fillWidth: true
            }

            DeviceSelectionList {
                id: deviceSelectionList
                workspace: root.workspace
                height: Math.round(160 * Scaling.uiScale)
                Layout.fillWidth: true
            }

            DeviceControlRow {
                id: deviceControlRow
                adapter: deviceSelectionList.selectedDevice
                Layout.fillWidth: true
            }

            // TODO
            // why is this not scrolling??

            ScrollView {
                id: deviceConfigScrollView
                Layout.fillWidth: true
                Layout.fillHeight: true

                Component.onCompleted: contentItem.boundsBehavior = Flickable.StopAtBounds
                clip: true

                // the container/listview pattern here makes sure each device configuration stays
                // in sync with our workspace model data.
                // Just using a single ConfigurationEditor that had a bound & dynamically updated
                // model wasn't able to refresh its own data automatically for some reason...
                // So the ListView of device config 'pages' just ensures the config UI is always 
                // updated in real time.

                Container {
                    id: deviceConfigContainer
                    anchors.fill: parent

                    contentItem: ListView {
                        id: deviceConfigListView
                        model: deviceConfigContainer.contentModel
                        snapMode: ListView.SnapOneItem
                        orientation: ListView.Horizontal
                        interactive: false

                        Connections {
                            target: deviceSelectionList
                            function onActivated(index) {
                                deviceConfigListView.positionViewAtIndex(index, ListView.SnapPosition);
                            }
                        }

                        Component.onCompleted: function () {
                            deviceConfigListView.currentIndex = root.workspace.selectedDeviceIndex;
                            deviceConfigListView.positionViewAtIndex(root.workspace.selectedDeviceIndex, ListView.SnapPosition);
                        }
                    }

                    Repeater {
                        model: root.workspace ? root.workspace.deviceAdapters : 0

                        ConfigurationEditor {
                            required property var modelData
                            configAdapter: modelData
                            workspace: root.workspace
                            flattenFields: false
                            width: deviceConfigContainer.width
                        }
                    }
                }
            }
        }
    }
}
