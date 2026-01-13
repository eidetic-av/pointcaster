import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

Item {
    id: root
    property var model: null

    property string emptyTitle: "Properties"
    property int maxBodyHeight: 320
    // shared width of the left 'label' column
    property int labelColumnWidth: WorkspaceState.labelColumnWidth
    property bool expanded: true

    implicitHeight: inspectorGroup.implicitHeight

    Rectangle {
        id: inspectorGroup
        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
            leftMargin: 8
            rightMargin: 8
            topMargin: 4
        }
        color: DarkPalette.dark
        border.color: DarkPalette.mid
        border.width: 1
        clip: true

        implicitHeight: header.height + (root.expanded ? (body.implicitHeight + body.anchors.topMargin) : 0)

        // header row (click to collapse/expand)
        Rectangle {
            id: header
            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
            }
            height: 28
            color: headerMouseArea.containsMouse ? DarkPalette.mid : root.expanded ? DarkPalette.middark : DarkPalette.base
            border.color: DarkPalette.mid
            border.width: 1

            Image {
                id: headerArrowIcon
                width: 12
                height: 12
                fillMode: Image.PreserveAspectFit
                source: root.expanded ? FontAwesome.icon("solid/caret-down") : FontAwesome.icon("solid/caret-right")
                opacity: 0.75
                anchors {
                    left: parent.left
                    leftMargin: 3
                    verticalCenter: parent.verticalCenter
                }
            }

            Text {
                id: headerText
                anchors {
                    left: headerArrowIcon.left
                    right: parent.right
                    leftMargin: 15
                    rightMargin: 10
                    verticalCenter: parent.verticalCenter
                }
                text: root.model ? (root.model.displayName() + " Properties") : root.emptyTitle
                elide: Text.ElideRight
                font.weight: Font.Medium
                font.pixelSize: 15
                color: DarkPalette.text
            }

            MouseArea {
                id: headerMouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: root.expanded = !root.expanded
            }
        }

        Item {
            id: body
            anchors {
                top: header.bottom
                left: parent.left
                right: parent.right
            }
            visible: root.expanded

            implicitHeight: {
                if (!root.expanded)
                    return 0;
                if (!root.model)
                    return noSelectionText.implicitHeight + 8;
                return Math.min(propertiesColumn.implicitHeight, root.maxBodyHeight);
            }

            Text {
                id: noSelectionText
                anchors {
                    left: parent.left
                    right: parent.right
                    leftMargin: 10
                    rightMargin: 10
                    top: parent.top
                }
                visible: !root.model
                text: "No device selected"
                font.weight: Font.Medium
                color: DarkPalette.text
            }

            Flickable {
                id: propertiesFlick
                anchors {
                    top: parent.top
                    left: parent.left
                    right: parent.right
                }
                height: Math.min(propertiesColumn.implicitHeight, root.maxBodyHeight)
                visible: root.model
                contentWidth: width
                contentHeight: propertiesColumn.implicitHeight
                clip: true

                ScrollBar.vertical: ScrollBar {
                    policy: ScrollBar.AsNeeded
                }

                Column {
                    id: propertiesColumn
                    width: propertiesFlick.width
                    spacing: 0

                    Repeater {
                        model: root.model ? root.model.fieldCount() : 0

                        delegate: Item {
                            required property int index

                            width: propertiesColumn.width
                            height: Math.max(nameText.implicitHeight, valueEditorLoader.implicitHeight) + 6
                            visible: root.model && !root.model.isHidden(index)

                            Item {
                                id: labelContainer
                                anchors {
                                    top: parent.top
                                    bottom: parent.bottom
                                    left: parent.left
                                    leftMargin: 7
                                }
                                width: root.labelColumnWidth

                                Text {
                                    id: nameText
                                    anchors {
                                        left: parent.left
                                        right: parent.right
                                        rightMargin: 10
                                        verticalCenter: parent.verticalCenter
                                    }
                                    horizontalAlignment: Text.AlignLeft
                                    color: DarkPalette.text
                                    text: root.model ? root.model.fieldName(index) : ""
                                    font.pixelSize: 15
                                    elide: Text.ElideRight
                                }
                            }

                            Loader {
                                id: valueEditorLoader
                                anchors {
                                    left: labelContainer.right
                                    leftMargin: -5
                                    right: parent.right
                                    top: parent.top
                                    bottom: parent.bottom
                                }
                                z: 3

                                readonly property string typeName: root.model ? String(root.model.fieldTypeName(index)).toLowerCase() : ""

                                sourceComponent: {
                                    if (typeName === "int" || typeName === "int32" || typeName === "int32_t" || typeName === "integer")
                                        return intEditor;
                                    if (typeName === "float" || typeName === "float32" || typeName === "float32_t" || typeName === "double" || typeName === "real" || typeName === "number")
                                        return floatEditor;
                                    return stringEditor;
                                }
                            }

                            Component {
                                id: stringEditor

                                TextField {
                                    id: valueField
                                    background: Rectangle {
                                        color: "transparent"
                                        border.color: valueField.focus ? DarkPalette.highlight : "transparent"
                                        border.width: 1
                                        radius: 0
                                    }
                                    font.pixelSize: 15
                                    text: root.model ? String(root.model.fieldValue(index)) : ""
                                    enabled: root.model ? !root.model.isDisabled(index) : false
                                    opacity: enabled ? 1.0 : 0.66

                                    onEditingFinished: {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, text);
                                    }
                                }
                            }

                            Component {
                                id: intEditor

                                DragInt {
                                    id: valueSpin

                                    boundEnabled: root.model ? !root.model.isDisabled(index) : false

                                    minValue: root.model ? root.model.minMax(index)[0] : undefined
                                    maxValue: root.model ? root.model.minMax(index)[1] : undefined
                                    defaultValue: root.model ? root.model.defaultValue(index) : undefined

                                    boundValue: {
                                        if (!root.model)
                                            return 0;
                                        var n = Number(root.model.fieldValue(index));
                                        return isNaN(n) ? 0 : Math.trunc(n);
                                    }

                                    onCommitValue: v => {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, v);
                                    }
                                }
                            }

                            Component {
                                id: floatEditor

                                DragFloat {
                                    id: valueFloat

                                    boundEnabled: root.model ? !root.model.isDisabled(index) : false

                                    minValue: root.model ? root.model.minMax(index)[0] : undefined
                                    maxValue: root.model ? root.model.minMax(index)[1] : undefined
                                    defaultValue: root.model ? root.model.defaultValue(index) : undefined

                                    boundValue: {
                                        if (!root.model)
                                            return 0.0;
                                        var n = Number(root.model.fieldValue(index));
                                        return isNaN(n) ? 0.0 : n;
                                    }

                                    onCommitValue: v => {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index);
                                    }
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    id: dividerHandle
                    x: root.labelColumnWidth - width / 2
                    z: 2
                    width: 5
                    anchors {
                        top: parent.top
                        bottom: parent.bottom
                        bottomMargin: 1
                    }
                    color: "transparent"

                    Rectangle {
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.horizontalCenter: parent.horizontalCenter
                        height: parent.height
                        width: (dividerMouseArea.containsMouse || dividerMouseArea.drag.active) ? 3 : 1
                        Behavior on width {
                            NumberAnimation {
                                duration: 120
                                easing.type: Easing.InCubic
                            }
                        }
                        color: dividerMouseArea.drag.active ? DarkPalette.midlight : dividerMouseArea.containsMouse ? DarkPalette.mid : DarkPalette.middark
                        Behavior on color {
                            ColorAnimation {
                                duration: 120
                                easing.type: Easing.InCubic
                            }
                        }
                    }

                    MouseArea {
                        id: dividerMouseArea
                        anchors.fill: parent
                        cursorShape: Qt.SplitHCursor
                        hoverEnabled: true
                        drag.target: dividerHandle
                        drag.axis: Drag.XAxis
                        drag.minimumX: 80
                        drag.maximumX: root.width - 80
                        onPositionChanged: {
                            WorkspaceState.labelColumnWidth = dividerHandle.x + dividerHandle.width / 2;
                        }
                    }
                }
            }
        }
    }
}
