import QtQuick
import QtQuick.Controls
import QtQml.Models
import com.kdab.dockwidgets as KDDW

import Pointcaster.Workspace 1.0

KDDW.DockWidget {
    uniqueName: "devicesWindow"
    title: "Devices"

    property var workspace: null

    Item {
        id: devices
        anchors.fill: parent
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

        // controller for the currently selected device (index-aligned with adapters)
        property var currentController: (workspace && deviceSelectionList.currentIndex >= 0 && deviceSelectionList.currentIndex < workspace.deviceControllers.length) ? workspace.deviceControllers[deviceSelectionList.currentIndex] : null

        Rectangle {
            id: background
            anchors.fill: parent
            color: DarkPalette.base
        }

        DeviceSelectionList {
            id: deviceSelectionList
            deviceAdapters: workspace ? workspace.deviceConfigAdapters : null
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.topMargin: 10
            anchors.leftMargin: 10
            anchors.rightMargin: 10
            height: 160
        }

        Row {
            id: controlRow
            anchors {
                top: deviceSelectionList.bottom
                left: parent.left
                right: parent.right
                leftMargin: 8
                rightMargin: 8
                topMargin: 4
            }
            height: 40
            spacing: 8

            enabled: devices.currentController !== null

            Button {
                id: startStopButton
                text: {
                    if (!devices.currentController)
                        return "Start";
                    return devices.currentController.status === DevicePluginController.Active ? "Stop" : "Start";
                }

                enabled: {
                    if (!devices.currentController)
                        return false;
                    // Disable interaction if the device is "Missing"
                    return devices.currentController.status !== DevicePluginController.Missing;
                }

                onClicked: {
                    if (!devices.currentController)
                        return;
                    if (devices.currentController.status === DevicePluginController.Active)
                        devices.currentController.stop();
                    else
                        devices.currentController.start();
                }
            }

            Button {
                id: restartButton
                text: "Restart"
                enabled: devices.currentController && devices.currentController.status === DevicePluginController.Active

                onClicked: {
                    if (devices.currentController)
                        devices.currentController.restart();
                }
            }

            // --- CPU/GPU exclusive toggle ---
            Item {
                id: computeToggle
                height: parent.height
                width: toggleRow.implicitWidth

                property bool cpuSelected: true

                Row {
                    id: toggleRow
                    spacing: 0

                    Rectangle {
                        id: cpuSeg
                        width: 40
                        height: 24

                        border.width: 1
                        border.color: DarkPalette.mid

                        color: DarkPalette.dark
                        z: computeToggle.cpuSelected ? 2 : 1

                        Text {
                            id: cpuLabel
                            text: "CPU"
                            anchors.left: parent.left
                            anchors.leftMargin: 5
                            anchors.top: parent.top
                            anchors.topMargin: 2
                            color: computeToggle.cpuSelected ? DarkPalette.text : DarkPalette.placeholderText
                            font.pixelSize: 11
                            font.weight: Font.Medium
                        }
                    }

                    Rectangle {
                        id: gpuSeg
                        width: 40
                        height: 24

                        border.width: 1
                        border.color: DarkPalette.mid

                        // shift left to overlap CPU button border
                        x: -1

                        color: DarkPalette.dark
                        z: computeToggle.cpuSelected ? 1 : 2

                        Text {
                            id: gpuLabel
                            text: "GPU"
                            anchors.right: parent.right
                            anchors.rightMargin: 5
                            anchors.top: parent.top
                            anchors.topMargin: 2
                            color: computeToggle.cpuSelected ? DarkPalette.placeholderText : DarkPalette.text
                            font.pixelSize: 11
                            font.weight: Font.Medium
                        }
                    }
                }

                // Single interaction surface
                MouseArea {
                    anchors.fill: toggleRow
                    cursorShape: Qt.PointingHandCursor

                    onClicked: {
                        computeToggle.cpuSelected = !computeToggle.cpuSelected;

                        // if (devices.currentController) {
                        //     devices.currentController.setComputeBackend(
                        //         computeToggle.cpuSelected
                        //             ? DevicePluginController.Cpu
                        //             : DevicePluginController.Gpu)
                        // }
                    }
                }
            }
        }

        // inspector for selected device
        Item {
            id: propertiesPane
            anchors {
                top: controlRow.bottom
                left: parent.left
                right: parent.right
            }
            anchors.topMargin: 6

            // shared width of the left "label" column
            property int labelColumnWidth: width * 0.35

            // collapsible group state
            property bool propertiesExpanded: true

            Rectangle {
                id: propertiesGroup
                anchors {
                    top: parent.top
                    left: parent.left
                    right: parent.right
                    leftMargin: 8
                    rightMargin: 8
                    topMargin: 4
                }
                color: DarkPalette.dark
                border.color: DarkPalette.mid
                border.width: 1
                clip: true

                height: groupHeader.height + (propertiesPane.propertiesExpanded ? (groupBody.implicitHeight + groupBody.anchors.topMargin) : 0)

                // header row (click to collapse/expand)
                Rectangle {
                    id: groupHeader
                    anchors {
                        top: parent.top
                        left: parent.left
                        right: parent.right
                    }
                    height: 28
                    color: DarkPalette.base
                    border.color: DarkPalette.mid
                    border.width: 1

                    Text {
                        id: groupHeaderText
                        anchors {
                            left: parent.left
                            right: parent.right
                            leftMargin: 10
                            rightMargin: 10
                            verticalCenter: parent.verticalCenter
                        }
                        text: devices.currentAdapter ? ("Device Properties — " + devices.currentAdapter.displayName()) : "Device Properties"
                        font.weight: Font.Medium
                        color: DarkPalette.text
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: propertiesPane.propertiesExpanded = !propertiesPane.propertiesExpanded
                    }
                }

                // body (properties list)
                Item {
                    id: groupBody
                    anchors {
                        top: groupHeader.bottom
                        left: parent.left
                        right: parent.right
                    }

                    visible: propertiesPane.propertiesExpanded

                    implicitHeight: {
                        if (!propertiesPane.propertiesExpanded)
                            return 0;

                        if (!devices.currentAdapter)
                            return noDeviceText.implicitHeight + 8;

                        // header spacing inside groupBody + scrollable content height
                        return Math.min(propertiesColumn.implicitHeight, 320);
                    }

                    // empty state when no selection
                    Text {
                        id: noDeviceText
                        anchors {
                            left: parent.left
                            right: parent.right
                            leftMargin: 10
                            rightMargin: 10
                            top: parent.top
                        }
                        visible: !devices.currentAdapter
                        text: "No device selected"
                        font.weight: Font.Medium
                        color: DarkPalette.text
                    }

                    Flickable {
                        id: propertiesFlick
                        anchors {
                            top: parent.top
                            left: parent.left
                            right: parent.right
                        }
                        height: Math.min(propertiesColumn.implicitHeight, 320)

                        visible: devices.currentAdapter

                        contentWidth: width
                        contentHeight: propertiesColumn.implicitHeight
                        clip: true

                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded
                        }

                        // Property rows
                        Column {
                            id: propertiesColumn
                            width: propertiesFlick.width
                            spacing: 4

                            Repeater {
                                model: devices.currentAdapter ? devices.currentAdapter.fieldCount() : 0

                                delegate: Item {
                                    required property int index

                                    width: propertiesColumn.width
                                    height: Math.max(nameText.implicitHeight, valueField.implicitHeight) + 6

                                    visible: !devices.currentAdapter.isHidden(index)

                                    // Label region (left column)
                                    Item {
                                        id: labelContainer
                                        anchors {
                                            left: parent.left
                                            top: parent.top
                                            bottom: parent.bottom
                                        }
                                        width: propertiesPane.labelColumnWidth

                                        Text {
                                            id: nameText
                                            anchors {
                                                left: parent.left
                                                right: parent.right
                                                verticalCenter: parent.verticalCenter
                                            }
                                            horizontalAlignment: Text.AlignLeft
                                            color: DarkPalette.text
                                            text: devices.currentAdapter.fieldName(index)
                                        }
                                    }

                                    // value region (right column)
                                    TextField {
                                        id: valueField
                                        anchors {
                                            left: labelContainer.right
                                            right: parent.right
                                            verticalCenter: parent.verticalCenter
                                        }
                                        text: devices.currentAdapter.fieldValue(index)
                                        enabled: !devices.currentAdapter.isDisabled(index)

                                        onEditingFinished: {
                                            devices.currentAdapter.setFieldValue(index, text);
                                        }
                                    }
                                }
                            }
                        }

                        // vertical divider handle (two-layer)
                        Rectangle {
                            id: dividerHandle
                            x: propertiesPane.labelColumnWidth - width / 2
                            width: 4
                            anchors {
                                top: parent.top
                                bottom: parent.bottom
                                bottomMargin: 1
                            }
                            color: DarkPalette.base

                            Rectangle {
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: 1
                                height: parent.height
                                color: DarkPalette.mid
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.SplitHCursor
                                drag.target: dividerHandle
                                drag.axis: Drag.XAxis
                                drag.minimumX: 80
                                drag.maximumX: propertiesPane.width - 80

                                onPositionChanged: {
                                    propertiesPane.labelColumnWidth = dividerHandle.x + dividerHandle.width / 2;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
