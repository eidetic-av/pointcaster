import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQml.Models

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

ToolBar {
    id: root

    property var workspace: null

    font: Scaling.uiFont

    anchors {
        top: parent.top
        left: parent.left
        right: parent.right
        topMargin: Math.round(10 * Scaling.uiScale)
        leftMargin: Math.round(8 * Scaling.uiScale)
        rightMargin: Math.round(8 * Scaling.uiScale)
    }

    background: Rectangle {
        color: "transparent"
        border.width: 0
        radius: Math.round(4 * Scaling.uiScale)
    }

    contentItem: RowLayout {
        spacing: Math.round(6 * Scaling.uiScale)

        ToolButton {
            id: addButton

            Layout.preferredWidth: Math.round(25 * Scaling.uiScale)
            Layout.preferredHeight: Math.round(25 * Scaling.uiScale)

            icon.source: FontAwesome.icon("solid/plus")
            icon.width: Math.round(18 * Scaling.uiScale)
            icon.height: Math.round(18 * Scaling.uiScale)

            InfoToolTip {
                visible: parent.hovered && !addDeviceMenu.visible
                textValue: "Add new device"
            }

            checkable: true
            checked: addDeviceMenu.visible

            onClicked: {
                addDeviceMenu.popup(addButton.x, addButton.y + Math.round(23 * Scaling.uiScale));
                if (root.workspace)
                    root.workspace.triggerDeviceDiscovery();
            }

            Menu {
                id: addDeviceMenu
                width: Math.round(220 * Scaling.uiScale)

                Instantiator {
                    model: root.workspace ? root.workspace.addDeviceMenuEntries : []

                    delegate: MenuItem {
                        required property var modelData

                        text: modelData.display_name + " (" + modelData.ip + ")"
                        font: root.font

                        contentItem: Text {
                            text: parent.text
                            color: DarkPalette.text
                            font: parent.font
                            elide: Text.ElideRight
                            verticalAlignment: Text.AlignVCenter
                        }

                        background: Rectangle {
                            color: hovered ? DarkPalette.mid : DarkPalette.almostdark
                            border.width: 1
                            border.color: DarkPalette.mid
                        }

                        onTriggered: {
                            addDeviceMenu.close();
                            if (modelData.kind === "discovered") {
                                workspace.addNewDevice(modelData.plugin_name, modelData.ip);
                            } else {
                                workspace.addNewDevice(modelData.plugin_name);
                            }
                        }
                    }

                    onObjectAdded: (i, o) => addDeviceMenu.insertItem(i, o)
                    onObjectRemoved: (i, o) => addDeviceMenu.removeItem(o)
                }
            }
        }

        ToolButton {
            Layout.preferredWidth: Math.round(25 * Scaling.uiScale)
            Layout.preferredHeight: Math.round(25 * Scaling.uiScale)

            icon.source: FontAwesome.icon("solid/trash")
            icon.width: Math.round(18 * Scaling.uiScale)
            icon.height: Math.round(18 * Scaling.uiScale)

            InfoToolTip {
                visible: tooltip ? parent.hovered : false
                textValue: tooltip
            }

            onClicked: {
                root.workspace.deleteSelectedDevice();
            }
        }

        Item {
            Layout.fillWidth: true
        }
    }
}
