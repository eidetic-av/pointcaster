import QtQuick
import QtQuick.Controls
import QtQml.Models
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

KDDW.DockWidget {

    uniqueName: "devicesWindow"
    title: "Devices"

    Item {
        id: devices
        anchors.fill: parent
        clip: true

        // Shared state for the properties pane
        property var currentAdapter: configList.currentItem ? configList.currentItem.modelData : null

        // Controller for the currently selected device (index-aligned with adapters)
        property var currentController: (workspaceModel
                                 && configList.currentIndex >= 0
                                 && configList.currentIndex < workspaceModel.deviceControllers.length)
                                ? workspaceModel.deviceControllers[configList.currentIndex]
                                : null


        // Top: device selector list
        ListView {
            id: configList
            model: workspaceModel ? workspaceModel.deviceConfigAdapters : []

            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
            }

            // Fixed height "selector" strip
            height: 160
            clip: true

            ScrollBar.vertical: ScrollBar {
                policy: ScrollBar.AsNeeded
            }

            // Remove kinetic "flick" behaviour
            boundsBehavior: Flickable.StopAtBounds
            maximumFlickVelocity: 0
            flickDeceleration: 1000000

            delegate: Rectangle {
                id: deviceRow
                required property int index
                required property ConfigAdapter modelData

                width: ListView.view.width
                height: 40

                color: ListView.isCurrentItem
                       ? Qt.rgba(0.2, 0.4, 0.8, 0.25)
                       : "transparent"

                border.color: "#404040"
                border.width: 0.5

                Row {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 8

                    Rectangle {
                        id: statusDot
                        width: 16
                        height: 16
                        radius: 8

                        color: {
                            switch (modelData.status) {
                            case UiEnums.WorkspaceDeviceStatus.Unloaded:
                                return "grey"
                            case UiEnums.WorkspaceDeviceStatus.Loaded:
                                return "steelblue"
                            case UiEnums.WorkspaceDeviceStatus.Active:
                                return "limegreen"
                            case UiEnums.WorkspaceDeviceStatus.Missing:
                                return "orangered"
                            default:
                                return "black"
                            }
                        }

                        border.width: 1
                        border.color: Qt.darker(color)
                    }

                    Text {
                        text: modelData.displayName()
                        verticalAlignment: Text.AlignVCenter
                        elide: Text.ElideRight
                    }

                    Text {
                        text: modelData.fieldValue(1) ? "active" : "inactive"
                        verticalAlignment: Text.AlignVCenter
                        color: modelData.fieldValue(1) ? "#6fb46f" : "#b46f6f"
                    }
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: configList.currentIndex = deviceRow.index
                }
            }
        }

        // *** NEW CONTROL BAR between list and inspector ***
        Row {
            id: controlRow
            anchors {
                top: configList.bottom
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
                        return "Start"
                    return devices.currentController.status === DevicePluginController.Active
                           ? "Stop"
                           : "Start"
                }

                enabled: {
                    if (!devices.currentController)
                        return false
                    // Disable interaction if the device is "Missing"
                    return devices.currentController.status !== DevicePluginController.Missing
                }

                onClicked: {
                    if (!devices.currentController)
                        return
                    if (devices.currentController.status === DevicePluginController.Active)
                        devices.currentController.stop()
                    else
                        devices.currentController.start()
                }
            }

            Button {
                id: restartButton
                text: "Restart"
                enabled: devices.currentController
                         && devices.currentController.status === DevicePluginController.Active

                onClicked: {
                    if (devices.currentController)
                        devices.currentController.restart()
                }
            }
        }

        // inspector for selected device
        Item {
            id: propertiesPane
            anchors {
                top: controlRow.bottom      // *** was configList.bottom ***
                left: parent.left
                right: parent.right
                bottom: parent.bottom
            }
            anchors.topMargin: 6

            // Shared width of the left "label" column
            property int labelColumnWidth: width * 0.35

            // Title / empty state
            Text {
                id: header
                anchors {
                    top: parent.top
                    left: parent.left
                    right: parent.right
                    leftMargin: 8
                    rightMargin: 8
                    topMargin: 4
                }
                text: devices.currentAdapter
                      ? "Device properties — " + devices.currentAdapter.displayName()
                      : "No device selected"
                font.bold: true
                elide: Text.ElideRight
            }

            Flickable {
                id: propertiesFlick
                anchors {
                    top: header.bottom
                    topMargin: 8
                    left: parent.left
                    right: parent.right
                    bottom: parent.bottom
                    leftMargin: 8
                    rightMargin: 8
                    bottomMargin: 8
                }

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
                            height: Math.max(nameText.implicitHeight,
                                             valueField.implicitHeight) + 6

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
                                    horizontalAlignment: Text.AlignRight
                                    elide: Text.ElideRight
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
                                    devices.currentAdapter.setFieldValue(index, text)
                                }
                            }
                        }
                    }
                }

                // vertical divider handle
                Rectangle {
                    id: dividerHandle
                    x: propertiesPane.labelColumnWidth - width / 2
                    width: 4
                    anchors {
                        top: parent.top
                        bottom: parent.bottom
                    }
                    color: "#505050"
                    opacity: 0.6

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.SplitHCursor
                        drag.target: dividerHandle
                        drag.axis: Drag.XAxis
                        drag.minimumX: 80
                        drag.maximumX: propertiesPane.width - 80

                        onPositionChanged: {
                            propertiesPane.labelColumnWidth =
                                    dividerHandle.x + dividerHandle.width / 2
                        }
                    }
                }
            }
        }
    }
}
