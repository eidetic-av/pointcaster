import QtQuick
import QtQuick.Controls

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    property var workspace: null
    property int currentIndex: list.currentIndex
    property var currentItem: list.currentItem

    signal activated(int index)

    height: Math.round(160 * Scaling.uiScale)
    anchors.left: parent.left
    anchors.right: parent.right

    Rectangle {
        id: frame
        anchors.fill: parent
        color: ThemeColors.alternateBase
        border.color: ThemeColors.mid
        border.width: 1
        clip: true

        ListView {
            id: list
            anchors.fill: parent
            anchors.margins: 1

            model: root.workspace.deviceConfigAdapters
            focus: true
            activeFocusOnTab: true
            keyNavigationEnabled: true
            boundsBehavior: Flickable.StopAtBounds

            ScrollBar.vertical: ScrollBar {
                policy: ScrollBar.AsNeeded
            }

            // keyboard: up/down + enter/space activates
            Keys.onPressed: event => {
                if (event.key === Qt.Key_Up) {
                    list.decrementCurrentIndex();
                    list.positionViewAtIndex(list.currentIndex, ListView.Contain);
                    event.accepted = true;
                } else if (event.key === Qt.Key_Down) {
                    list.incrementCurrentIndex();
                    list.positionViewAtIndex(list.currentIndex, ListView.Contain);
                    event.accepted = true;
                } else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter || event.key === Qt.Key_Space) {
                    if (list.currentIndex >= 0) {
                        root.activated(list.currentIndex);
                        root.workspace.setSelectedDeviceIndex(list.currentIndex);
                    }
                    event.accepted = true;
                }
            }

            delegate: Rectangle {
                id: row
                width: list.width
                height: Math.max(Math.round(30 * Scaling.uiScale), Math.ceil(Scaling.pointSize * 2.1))

                property bool hovered: mouse.containsMouse
                property bool selected: ListView.isCurrentItem

                color: hovered ? ThemeColors.midlight : selected ? ThemeColors.mid : ThemeColors.almostdark
                border.width: 0

                Row {
                    anchors.fill: parent
                    anchors.leftMargin: Math.round(8 * Scaling.uiScale)
                    anchors.rightMargin: Math.round(8 * Scaling.uiScale)
                    spacing: Math.round(8 * Scaling.uiScale)

                    Rectangle {
                        id: statusCircle
                        width: Math.round(10 * Scaling.uiScale)
                        height: Math.round(10 * Scaling.uiScale)
                        radius: Math.round(5 * Scaling.uiScale)
                        anchors.verticalCenter: parent.verticalCenter
                        color: {
                            if (!modelData || modelData.status === undefined)
                                return ThemeColors.inactive;
                            switch (modelData.status) {
                            case UiEnums.WorkspaceDeviceStatus.Loaded:
                                return ThemeColors.neutralSuccess;
                            case UiEnums.WorkspaceDeviceStatus.Active:
                                return ThemeColors.success;
                            case UiEnums.WorkspaceDeviceStatus.Missing:
                                return ThemeColors.error;
                            default:
                                return ThemeColors.inactive;
                            }
                        }
                    }

                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        width: Math.max(Math.round(75 * Scaling.uiScale), parent.width - statusCircle.width - deviceTypeText.width - (parent.spacing * 2))
                        elide: Text.ElideRight
                        text: modelData.fieldValue(0)
                        color: ThemeColors.text
                        font: Scaling.uiFont
                    }

                    Text {
                        id: deviceTypeText
                        anchors.verticalCenter: parent.verticalCenter
                        elide: Text.ElideRight
                        text: modelData.displayName()
                        color: ThemeColors.text
                        opacity: .5
                        font: Scaling.uiFont
                    }
                }

                MouseArea {
                    id: mouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        list.forceActiveFocus();
                        list.currentIndex = index;
                        root.activated(index);
                        root.workspace.setSelectedDeviceIndex(list.currentIndex);
                    }
                }
            }
        }

        // focus ring
        Rectangle {
            anchors.fill: parent
            anchors.margins: 1
            radius: frame.radius
            color: "transparent"
            border.width: list.activeFocus ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0
            border.color: ThemeColors.highlight
            visible: border.width > 0
        }
    }
}
