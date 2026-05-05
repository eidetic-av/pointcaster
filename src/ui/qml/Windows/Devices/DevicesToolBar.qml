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

        Item {
            Layout.fillWidth: true
            Layout.minimumWidth: 0

            implicitHeight: label.implicitHeight + underline.height

            TextMetrics {
                id: textMetrics
                font: label.font
                text: label.text
            }

            Label {
                id: label
                text: "session_1"

                font: root.font
                color: ThemeColors.text
                opacity: 0.9

                elide: Text.ElideRight
                verticalAlignment: Text.AlignVCenter
                topPadding: Math.round(3 * Scaling.uiScale)

                anchors.left: parent.left
                anchors.right: parent.right
            }

            Rectangle {
                id: underline
                height: 1
                width: Math.min(textMetrics.width, label.width)   // respects eliding
                color: ThemeColors.highlight

                anchors {
                    left: label.left
                    top: label.bottom
                    topMargin: Math.round(1 * Scaling.uiScale)
                }
            }
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
            Layout.preferredWidth: Math.round(25 * Scaling.uiScale)
            Layout.preferredHeight: Math.round(25 * Scaling.uiScale)

            icon.source: FontAwesome.icon("solid/trash")
            icon.width: Math.round(17 * Scaling.uiScale)
            icon.height: Math.round(17 * Scaling.uiScale)

            enabled: root.workspace && root.workspace.deviceAdapters.length > 0
            opacity: enabled ? 1.0 : 0.4

            onClicked: {
                if (root.workspace)
                    root.workspace.deleteSelectedDevice();
            }

            InfoToolTip {
                visible: parent.hovered
                textValue: "Delete selected device"
            }
        }
    }
}
