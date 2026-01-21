import QtQuick
import QtQuick.Controls
import QtQml.Models
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0
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
            id: background
            anchors.fill: parent
            color: DarkPalette.base
        }

        Row {
            id: devicesButtonsRow
            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
                topMargin: 10
                leftMargin: 8
                rightMargin: 8
            }
            spacing: 8

            IconButton {
                tooltip: "Add new device"
                iconSource: FontAwesome.icon('solid/plus')
                onClicked: {
                    console.log('add device')
                }
            }

            IconButton {
                tooltip: "Delete selected device"
                iconSource: FontAwesome.icon('solid/trash')
                onClicked: {
                    console.log('delete device')
                }
            }

        }


        DeviceSelectionList {
            id: deviceSelectionList
            deviceAdapters: workspace ? workspace.deviceConfigAdapters : null
            anchors.top: devicesButtonsRow.bottom
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
                text: "Restart"
                iconSource: FontAwesome.icon("solid/rotate-right") // or "redo" depending on your FA set

                enabled: devices.currentController && devices.currentController.status === DevicePluginController.Active

                onClicked: {
                    if (devices.currentController)
                        devices.currentController.restart();
                }
            }

            // Item {
            //     id: computeToggle
            //     height: parent.height
            //     width: toggleRow.implicitWidth

            //     property bool cpuSelected: true

            //     Row {
            //         id: toggleRow
            //         spacing: 0

            //         Rectangle {
            //             id: cpuSeg
            //             width: 40
            //             height: 24

            //             border.width: 1
            //             border.color: DarkPalette.mid

            //             color: DarkPalette.dark
            //             z: computeToggle.cpuSelected ? 2 : 1

            //             Text {
            //                 text: "CPU"
            //                 anchors.left: parent.left
            //                 anchors.leftMargin: 5
            //                 anchors.top: parent.top
            //                 anchors.topMargin: 2
            //                 color: computeToggle.cpuSelected ? DarkPalette.text : DarkPalette.placeholderText
            //                 font.pixelSize: 11
            //                 font.weight: Font.Medium
            //             }
            //         }

            //         Rectangle {
            //             id: gpuSeg
            //             width: 40
            //             height: 24

            //             border.width: 1
            //             border.color: DarkPalette.mid
            //             x: -1

            //             color: DarkPalette.dark
            //             z: computeToggle.cpuSelected ? 1 : 2

            //             Text {
            //                 text: "GPU"
            //                 anchors.right: parent.right
            //                 anchors.rightMargin: 5
            //                 anchors.top: parent.top
            //                 anchors.topMargin: 2
            //                 color: computeToggle.cpuSelected ? DarkPalette.placeholderText : DarkPalette.text
            //                 font.pixelSize: 11
            //                 font.weight: Font.Medium
            //             }
            //         }
            //     }

            //     MouseArea {
            //         anchors.fill: toggleRow
            //         onClicked: {
            //             computeToggle.cpuSelected = !computeToggle.cpuSelected;

            //             // if (devices.currentController) {
            //             //     devices.currentController.setComputeBackend(
            //             //         computeToggle.cpuSelected
            //             //             ? DevicePluginController.Cpu
            //             //             : DevicePluginController.Gpu)
            //             // }
            //         }
            //     }
            // }
        }

        ConfigurationEditor {
            id: propertiesPane
            anchors {
                top: controlRow.bottom
                left: parent.left
                right: parent.right
            }
            anchors.topMargin: 6

            model: devices.currentAdapter
        }

        ConfigurationEditor {
            id: propertiesPane2
            anchors {
                top: propertiesPane.bottom
                left: parent.left
                right: parent.right
            }
            anchors.topMargin: 6

            model: devices.otherAdapter
        }
    }
}
