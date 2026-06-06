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
            deviceSelectionList.selectDevice(workspace.selectedDeviceIndex);
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
            deviceSelectionList.selectDevice(workspace.deviceAdapters.length - 1);
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
                id: deviceListContainer

                Layout.fillWidth: true
                Layout.preferredHeight: Math.round(Workspace.deviceListHeight * Scaling.uiScale)

                DeviceSelectionList {
                    id: deviceSelectionList
                    workspace: root.workspace

                    anchors {
                        top: parent.top
                        left: parent.left
                        right: parent.right
                        bottom: resizeHandle.top
                    }

                    onActivated: function (index) {
                        root.currentDeviceIndex = index;
                        root.deviceSelected();
                    }
                }

                TapHandler {
                    onTapped: root.workspace.selectedOperatorAdapter = null
                }

                Rectangle {
                    id: resizeHandle

                    anchors {
                        left: parent.left
                        right: parent.right
                        bottom: parent.bottom
                    }

                    height: Math.round(5 * Scaling.uiScale)
                    color: "transparent"

                    Rectangle {
                        anchors.centerIn: parent

                        width: parent.width
                        height: (resizeMouseArea.containsMouse || resizeMouseArea.dragging) ? Math.round(3 * Scaling.uiScale) : Math.max(1, Math.round(1 * Scaling.uiScale))

                        color: resizeMouseArea.dragging ? ThemeColors.mid : (resizeMouseArea.containsMouse ? ThemeColors.middark : ThemeColors.almostdark)

                        Behavior on height {
                            NumberAnimation {
                                duration: Math.round(120 * Scaling.uiScale)
                                easing.type: Easing.InCubic
                            }
                        }

                        Behavior on color {
                            ColorAnimation {
                                duration: Math.round(90 * Scaling.uiScale)
                                easing.type: Easing.InCubic
                            }
                        }
                    }

                    MouseArea {
                        id: resizeMouseArea

                        anchors.fill: parent

                        cursorShape: Qt.SizeVerCursor
                        hoverEnabled: true

                        property bool dragging: false

                        onPressed: dragging = true
                        onReleased: dragging = false
                        onCanceled: dragging = false

                        onPositionChanged: {
                            if (!dragging)
                                return;

                            Workspace.deviceListHeight = Math.max((80 * Scaling.uiScale), Math.round(deviceListContainer.height + mouseY));
                        }
                    }
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
                Layout.minimumHeight: Math.round(140 * Scaling.uiScale)
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
                                pipelineAdapter: deviceDelegate.modelData && deviceDelegate.modelData.operator_pipelineAdapter ? deviceDelegate.modelData.operator_pipelineAdapter : []
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
