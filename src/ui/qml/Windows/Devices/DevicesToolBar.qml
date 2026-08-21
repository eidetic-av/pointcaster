import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQml.Models

import Pointcaster 1.0

ToolBar {
    id: root

    property var workspace: null

    font: Scaling.uiFont

    width: parent.width

    anchors {
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

        SelectedSessionLabel {
            id: sessionLabel
            workspace: root.workspace
            Layout.fillWidth: true
            Layout.minimumWidth: 0
        }

        Item {
            visible: !sessionLabel.visible
            Layout.fillWidth: true
        }

        ToolButton {
            id: addButton

            Layout.preferredWidth: Math.round(25 * Scaling.uiScale)
            Layout.preferredHeight: Math.round(25 * Scaling.uiScale)

            icon.source: FontAwesome.icon("solid/plus")
            icon.width: Math.round(17 * Scaling.uiScale)
            icon.height: Math.round(17 * Scaling.uiScale)

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
        }

        AddDeviceMenu {
            id: addDeviceMenu
            workspace: root.workspace
        }

        ToolButton {
            id: addGroupButton

            Layout.preferredWidth: Math.round(25 * Scaling.uiScale)
            Layout.preferredHeight: Math.round(25 * Scaling.uiScale)

            icon.source: FontAwesome.icon("solid/folder-plus")
            icon.width: Math.round(22 * Scaling.uiScale)
            icon.height: Math.round(22 * Scaling.uiScale)

            onClicked: {
                if (root.workspace) root.workspace.createDeviceGroup();
            }

            InfoToolTip {
                visible: parent.hovered
                textValue: "Add new device group"
            }
        }

        ToolButton {
            id: duplicateButton

            Layout.preferredWidth: Math.round(25 * Scaling.uiScale)
            Layout.preferredHeight: Math.round(25 * Scaling.uiScale)

            icon.source: FontAwesome.icon("solid/clone")
            icon.width: Math.round(20 * Scaling.uiScale)
            icon.height: Math.round(20 * Scaling.uiScale)

            enabled: root.workspace && root.workspace.selectedNodeId != ""
            opacity: enabled ? 1.0 : 0.4

            onClicked: {
                if (!root.workspace) return;
                root.workspace.duplicateSelectedDeviceNode();
            }

            InfoToolTip {
                visible: parent.hovered
                textValue: "Duplicate selected device"
            }
        }

        ToolButton {
            Layout.preferredWidth: Math.round(25 * Scaling.uiScale)
            Layout.preferredHeight: Math.round(25 * Scaling.uiScale)

            icon.source: FontAwesome.icon("solid/trash")
            icon.width: Math.round(17 * Scaling.uiScale)
            icon.height: Math.round(17 * Scaling.uiScale)

            enabled: root.workspace && root.workspace.selectedNodeId != ""
            opacity: enabled ? 1.0 : 0.4

            onClicked: {
                if (root.workspace)
                    root.workspace.deleteSelectedDeviceNode();
            }

            InfoToolTip {
                visible: parent.hovered
                textValue: "Delete selected device"
            }
        }
    }
}
