import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

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
            onClicked: {
                list.forceActiveFocus();
                setSelectedIndex(index);
            }

            height: content.height
            width: content.width

            Rectangle {
                id: content

                property bool selected: list.currentIndex === index
                property bool hovered: hoverHandler.hovered

                HoverHandler {
                    id: hoverHandler
                }

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

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: Math.round(8 * Scaling.uiScale)
                    anchors.rightMargin: Math.round(8 * Scaling.uiScale)
                    spacing: Math.round(8 * Scaling.uiScale)

                    CheckBox {
                        id: activeToggle
                        checked: modelData.active
                        onToggled: modelData.active = checked
                        Layout.maximumWidth: Math.round(16 * Scaling.uiScale)
                        Layout.minimumWidth: Math.round(16 * Scaling.uiScale)
                        height: parent.height

                        indicator: Image {
                            anchors.centerIn: parent
                            source: activeToggle.checked ? FontAwesome.icon("solid/toggle-on") : FontAwesome.icon("solid/toggle-off")
                            sourceSize: Qt.size(9.5 * Scaling.uiScale, 9.5 * Scaling.uiScale)
                            opacity: activeToggle.hovered ? 0.7 : 0.5
                        }

                        InfoToolTip {
                            textValue: "Toggle device active"
                        }
                    }

                    CheckBox {
                        id: renderToggle
                        checked: modelData.render
                        onToggled: modelData.render = checked
                        Layout.maximumWidth: Math.round(16 * Scaling.uiScale)
                        Layout.minimumWidth: Math.round(16 * Scaling.uiScale)
                        height: parent.height

                        indicator: Image {
                            anchors.centerIn: parent
                            source: renderToggle.checked ? FontAwesome.icon("solid/eye") : FontAwesome.icon("solid/eye-slash")
                            sourceSize: Qt.size(9.5 * Scaling.uiScale, 9.5 * Scaling.uiScale)
                            opacity: renderToggle.hovered ? 0.7 : 0.5
                        }

                        InfoToolTip {
                            textValue: "Toggle session rendering"
                        }
                    }

                    Rectangle {
                        id: statusCircle
                        width: 7 * Scaling.uiScale
                        height: 7 * Scaling.uiScale
                        radius: 3.5 * Scaling.uiScale
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

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "transparent"

                        border.color: deviceLabelTextEdit.focus ? ThemeColors.highlight : "transparent"
                        border.width: 1

                        property string _startEditLabel

                        TextInput {
                            id: deviceLabelTextEdit
                            anchors.fill: parent
                            verticalAlignment: TextEdit.AlignVCenter
                            focus: false

                            color: ThemeColors.text
                            selectionColor: ThemeColors.highlight
                            selectedTextColor: ThemeColors.highlightedText
                            font: Scaling.uiFont

                            opacity: text == modelData.label ? 1 : 0.5

                            Component.onCompleted: {
                                text =  modelData.label || modelData.id
                            }

                            function focusForEdit() {
                                deviceLabelTextEdit.focus = true;
                                deviceLabelTextEdit.cursorVisible = true;
                                deviceLabelTextEdit.selectAll();
                                deviceLabelDoubleClickArea.visible = false;
                                parent._startEditLabel = deviceLabelTextEdit.text;
                            }

                            onEditingFinished: {
                                // TODO
                                // validate here the label isn't the same as any other
                                deviceLabelTextEdit.focus = false;
                                deviceLabelDoubleClickArea.visible = true;
                                if (text == parent._startEditLabel) return;
                                modelData.label = text
                                text = modelData.label || modelData.id
                            }
                        }

                        MouseArea {
                            id: deviceLabelDoubleClickArea
                            anchors.fill: parent
                            propagateComposedEvents: true
                            onClicked: { mouse.accepted = false }
                            onDoubleClicked: deviceLabelTextEdit.focusForEdit()
                        }
                    }


                    Text {
                        id: deviceTypeText
                        elide: Text.ElideRight
                        text: modelData.displayName() + (modelData.pluginNullState ? " (Unloaded)" : "")
                        color: ThemeColors.text
                        opacity: modelData.pluginNullState ? .25 : .5
                        font: Scaling.uiFont
                        Layout.rightMargin: Scaling.uiScale * 2
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
