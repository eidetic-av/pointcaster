import QtQml.Models
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

KDDW.DockWidget {
    id: root
    uniqueName: "devicesWindow"
    title: "Devices"

    property var workspace: null
    property int currentDeviceIndex: workspace ? workspace.selectedDeviceIndex : 0

    signal deviceSelected

    function init() {
        if (workspace) {
            workspace.triggerDeviceDiscovery();
            deviceSelectionList.setSelectedIndex(workspace.selectedDeviceIndex);
            currentDeviceIndex = workspace.selectedDeviceIndex;
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
            currentDeviceIndex = workspace.deviceAdapters.length - 1;
        }
    }

    // reset scroll position when switching devices
    onCurrentDeviceIndexChanged: {
        if (deviceConfigScrollView.contentItem)
            deviceConfigScrollView.contentItem.contentY = 0;
    }

    Item {
        anchors.fill: parent

        Rectangle {
            id: background
            anchors.fill: parent
            color: ThemeColors.base
        }

        ColumnLayout {
            id: innerContent

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

            Item {
                Layout.fillWidth: true
                Layout.preferredHeight: Math.round(160 * Scaling.uiScale)

                DeviceSelectionList {
                    id: deviceSelectionList
                    workspace: root.workspace
                    anchors.fill: parent
                    onActivated: function (index) {
                        root.currentDeviceIndex = index;
                        root.deviceSelected();
                    }
                }

                TapHandler {
                    onTapped: root.workspace.selectedOperatorAdapter = null
                }
            }

            DeviceControlRow {
                id: deviceControlRow
                adapter: deviceSelectionList.selectedDevice
                Layout.fillWidth: true
            }

            Timeline {
                id: sequenceTimeline
                adapter: deviceSelectionList.selectedDevice
                Layout.fillWidth: true
            }

            ScrollView {
                id: deviceConfigScrollView
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true

                Component.onCompleted: contentItem.boundsBehavior = Flickable.StopAtBounds

                Column {
                    width: deviceConfigScrollView.availableWidth
                    anchors.topMargin: Math.round(8 * Scaling.uiScale)

                    Repeater {
                        model: root.workspace ? root.workspace.deviceAdapters : 0

                        Column {
                            id: deviceDelegate
                            required property var modelData
                            required property int index
                            width: parent.width
                            visible: index === root.currentDeviceIndex

                            ConfigurationEditor {
                                id: deviceConfigEditor
                                configAdapter: deviceDelegate.modelData
                                workspace: root.workspace
                                flattenFields: false
                                width: parent.width
                            }

                            spacing: deviceConfigEditor.groupSpacing

                            OperatorPipelineEditor {
                                workspace: root.workspace
                                operators: deviceDelegate.modelData && deviceDelegate.modelData.operatorAdapters ? deviceDelegate.modelData.operatorAdapters : []
                                width: parent.width

                                onAddOperatorRequested: operatorType => root.workspace.addOperatorToDevice(deviceDelegate.index, operatorType)
                                onRemoveOperatorRequested: operatorIndex => root.workspace.removeOperatorFromDevice(deviceDelegate.index, operatorIndex)
                            }
                        }
                    }
                }
            }
        }
    }
}