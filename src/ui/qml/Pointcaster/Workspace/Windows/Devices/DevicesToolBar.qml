import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQml.Models

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

ToolBar {
    id: root

    property var workspace: null

    anchors {
        top: parent.top
        left: parent.left
        right: parent.right
        topMargin: 10
        leftMargin: 8
        rightMargin: 8
    }

    background: Rectangle {
        color: "transparent"
        border.width: 0
        radius: 4
    }

    contentItem: RowLayout {
        spacing: 6

        ToolButton {
            id: addButton

            Layout.preferredWidth: 25
            Layout.preferredHeight: 25

            icon.source: FontAwesome.icon("solid/plus")
            icon.width: 18
            icon.height: 18

            ToolTip.visible: hovered && !checked
            ToolTip.text: "Add new device"
            ToolTip.delay: 400

            checkable: true
            checked: addDeviceMenu.visible

            onClicked: {
                addDeviceMenu.popup(addButton.x, addButton.y + 23);
                if (root.workspace)
                    root.workspace.triggerDeviceDiscovery();
            }

            Menu {
                id: addDeviceMenu
                width: 220

                Instantiator {
                    model: root.workspace ? root.workspace.addDeviceMenuEntries : []

                    delegate: MenuItem {
                        required property var modelData

                        text: modelData.display_name + " (" + modelData.ip + ")"

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
            Layout.preferredWidth: 25
            Layout.preferredHeight: 25

            icon.source: FontAwesome.icon("solid/trash")
            icon.width: 18
            icon.height: 18

            ToolTip.visible: hovered
            ToolTip.text: "Delete selected device"
            ToolTip.delay: 400

            onClicked: {
                root.workspace.deleteSelectedDevice();
            }
        }

        Item {
            Layout.fillWidth: true
        }
    }
}
