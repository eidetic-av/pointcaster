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

    height: 160
    anchors.left: parent.left
    anchors.right: parent.right

    Rectangle {
        id: frame
        anchors.fill: parent
        color: DarkPalette.alternateBase
        border.color: DarkPalette.mid
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
                height: 30

                property bool hovered: mouse.containsMouse
                property bool selected: ListView.isCurrentItem

                color: hovered ? DarkPalette.midlight : selected ? DarkPalette.mid : DarkPalette.almostdark

                border.width: 0

                Row {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 8

                    Rectangle {
                        id: statusCircle
                        width: 10
                        height: 10
                        radius: 5
                        anchors.verticalCenter: parent.verticalCenter
                        color: {
                            if (!modelData || modelData.status === undefined)
                                return DarkPalette.inactive;
                            switch (modelData.status) {
                            case UiEnums.WorkspaceDeviceStatus.Loaded:
                                return DarkPalette.neutralSuccess;
                            case UiEnums.WorkspaceDeviceStatus.Active:
                                return DarkPalette.success;
                            case UiEnums.WorkspaceDeviceStatus.Missing:
                                return DarkPalette.error;
                            default:
                                return DarkPalette.inactive;
                            }
                        }
                    }

                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        width: Math.max(75, parent.width - statusCircle.width - deviceTypeText.width - (parent.spacing * 2))
                        elide: Text.ElideRight
                        text: modelData.fieldValue(0)
                        color: DarkPalette.text
                    }

                    Text {
                        id: deviceTypeText
                        anchors.verticalCenter: parent.verticalCenter
                        elide: Text.ElideRight
                        text: modelData.displayName()
                        color: DarkPalette.text
                        opacity: .5
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
            border.width: list.activeFocus ? 1 : 0
            border.color: DarkPalette.highlight
            visible: border.width > 0
        }
    }
}
