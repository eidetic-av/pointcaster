import QtQuick
import QtQuick.Controls

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    property var workspace: null
    property var currentItem: list.currentItem
    property var selectedDevice: null

    width: parent.width

    signal activated(int index)

    function setSelectedIndex(index) {
        list.currentIndex = index;
        workspace.selectedDeviceIndex = index;
        selectedDevice = workspace.deviceAdapters[index];
        activated(index);
    }

    Component {
        id: deviceDelegate

        MouseArea {
            id: dragArea

            property bool held: false

            drag.target: held ? content : undefined
            drag.axis: Drag.YAxis

            onPressAndHold: held = true
            onReleased: held = false

            height: content.height
            width: content.width

            Rectangle {
                id: content

                property bool selected: list.currentIndex === index
                property bool hovered: deviceRowMouseArea.containsMouse

                width: list.width
                height: Math.max(Math.round(30 * Scaling.uiScale), Math.ceil(Scaling.pointSize * 2.1))

                color: hovered ? ThemeColors.midlight : selected ? ThemeColors.mid : ThemeColors.almostdark

                border.width: 0

                states: State {
                    when: dragArea.held

                    ParentChange {
                        target: content
                        parent: root
                    }
                    AnchorChanges {
                        target: content
                        anchors {
                            horizontalCenter: undefined
                            verticalCenter: undefined
                        }
                    }
                }

                MouseArea {
                    id: deviceRowMouseArea
                    anchors.fill: parent
                    hoverEnabled: true
                    propagateComposedEvents: true
                    onClicked: {
                        list.forceActiveFocus();
                        setSelectedIndex(index);
                    }
                }

                Row {
                    anchors.fill: parent
                    anchors.leftMargin: Math.round(8 * Scaling.uiScale)
                    anchors.rightMargin: Math.round(8 * Scaling.uiScale)
                    spacing: Math.round(8 * Scaling.uiScale)

                    Rectangle {
                        id: statusCircle
                        width: Math.round(9 * Scaling.uiScale)
                        height: Math.round(9 * Scaling.uiScale)
                        radius: Math.round(5 * Scaling.uiScale)
                        anchors.verticalCenter: parent.verticalCenter
                        color: {
                            if (!modelData || modelData.status === undefined)
                                return ThemeColors.inactive;
                            if (modelData.pluginNullState)
                                return ThemeColors.error;
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

                    CheckBox {
                        id: renderCheckbox
                        checked: modelData.render
                        onToggled: modelData.render = checked
                    }

                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        width: Math.max(Math.round(75 * Scaling.uiScale), parent.width - statusCircle.width - deviceTypeText.width - (parent.spacing * 2))
                        elide: Text.ElideRight
                        text: modelData.id
                        color: ThemeColors.text
                        font: Scaling.uiFont
                    }

                    Text {
                        id: deviceTypeText
                        anchors.verticalCenter: parent.verticalCenter
                        elide: Text.ElideRight
                        text: modelData.displayName() + (modelData.pluginNullState ? " (Unloaded)" : "")
                        color: ThemeColors.text
                        opacity: modelData.pluginNullState ? .25 : .5
                        font: Scaling.uiFont
                    }
                }
            }
        }
    }

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

            model: root.workspace ? root.workspace.deviceAdapters : []
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
                        setSelectedIndex(list.currentIndex);
                    }
                    event.accepted = true;
                }
            }

            delegate: deviceDelegate
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
