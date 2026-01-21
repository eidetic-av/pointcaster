import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

Item {
    id: root
    property var model: null

    property string emptyTitle: "Properties"
    property int labelColumnWidth: WorkspaceState.labelColumnWidth
    property bool expanded: true

    property int outerHorizontalMargin: 8
    property int outerTopMargin: 4

    readonly property real _viewportWidth: {
        if (ScrollView.view) return ScrollView.view.width;
        if (parent) return parent.width;
        return 0;
    }
    width: _viewportWidth

    implicitHeight: root.outerTopMargin + inspectorGroup.implicitHeight

    Rectangle {
        id: inspectorGroup
        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
            leftMargin: root.outerHorizontalMargin
            rightMargin: root.outerHorizontalMargin
            topMargin: root.outerTopMargin
        }

        implicitHeight: header.height + (root.expanded ? body.implicitHeight : 0)

        color: DarkPalette.dark
        border.color: DarkPalette.mid
        border.width: 1
        clip: true

        Rectangle {
            id: header
            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
            }
            height: 28
            color: headerMouseArea.containsMouse ? DarkPalette.mid
                                                 : (root.expanded ? DarkPalette.middark : DarkPalette.base)
            border.color: DarkPalette.mid
            border.width: 1

            Image {
                id: headerArrowIcon
                width: 12
                height: 12
                fillMode: Image.PreserveAspectFit
                source: root.expanded ? FontAwesome.icon("solid/caret-down")
                                      : FontAwesome.icon("solid/caret-right")
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
                font.pixelSize: 13
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

            implicitHeight: root.model ? fieldsContainer.implicitHeight : (noSelectionText.implicitHeight + 8)

            Text {
                id: noSelectionText
                anchors {
                    left: parent.left
                    right: parent.right
                    top: parent.top
                    leftMargin: 10
                    rightMargin: 10
                    topMargin: 3
                }
                visible: !root.model
                text: "No device selected"
                font.weight: Font.Medium
                color: DarkPalette.text
            }

            Item {
                id: fieldsArea
                anchors {
                    top: parent.top
                    left: parent.left
                    right: parent.right
                    rightMargin: 1
                }
                visible: root.model
                height: fieldsContainer.implicitHeight

                Column {
                    id: fieldsContainer
                    width: fieldsArea.width
                    spacing: 0

                    Repeater {
                        model: root.model ? root.model.fieldCount() : 0

                        delegate: Item {
                            required property int index

                            width: fieldsContainer.width
                            height: Math.max(nameText.implicitHeight, valueContainer.implicitHeight) + 6
                            visible: root.model && !root.model.isHidden(index)

                            Item {
                                id: labelContainer
                                anchors {
                                    top: parent.top
                                    bottom: parent.bottom
                                    left: parent.left
                                    leftMargin: 6
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
                                    text: root.model ? StringUtils.titleFromSnake(root.model.fieldName(index)) : ""
                                    font.pixelSize: 13
                                    elide: Text.ElideRight
                                }

                                ToolTip {
                                    visible: labelHover.hovered
                                    text: root.model ? "osc/address/" + qsTr(root.model.fieldName(index)) : ""
                                    delay: 400
                                    parent: labelContainer
                                    x: Math.max((labelContainer.width - implicitWidth) * 0.5, 4)
                                    y: labelContainer.height + 4
                                }

                                HoverHandler { id: labelHover }
                            }

                            Loader {
                                id: valueContainer
                                anchors {
                                    left: labelContainer.right
                                    leftMargin: -5
                                    right: parent.right
                                    verticalCenter: parent.verticalCenter
                                }
                                z: 3

                                readonly property string typeName: root.model ? String(root.model.fieldTypeName(index)).toLowerCase() : ""

                                sourceComponent: {
                                    if (root.model && root.model.isEnum(index))
                                        return enumEditor;
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
                                    font.pixelSize: 13
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

                                    onCommitValue: function (v) {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, v);
                                    }
                                }
                            }

                            Component {
                                id: floatEditor
                                DragFloat {
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

                                    onCommitValue: function (v) {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, v);
                                    }
                                }
                            }

                            Component {
                                id: enumEditor
                                EnumSelector {
                                    id: enumSelector

                                    enabled: root.model ? !root.model.isDisabled(index) : false
                                    opacity: enabled ? 1.0 : 0.66

                                    options: root.model ? root.model.enumOptions(index) : []
                                    boundValue: {
                                        if (!root.model)
                                            return 0;
                                        var n = Number(root.model.fieldValue(index));
                                        return isNaN(n) ? 0 : Math.trunc(n);
                                    }

                                    onCommitValue: function (v) {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, v);
                                    }

                                    Connections {
                                        target: root.model
                                        function onFieldChanged(changedIndex) {
                                            if (changedIndex !== index)
                                                return;
                                            var n = Number(root.model.fieldValue(index));
                                            enumSelector.boundValue = isNaN(n) ? 0 : Math.trunc(n);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    id: dividerHandle
                    x: (root.labelColumnWidth - width / 2) + 1
                    z: 99
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
                            NumberAnimation { duration: 120; easing.type: Easing.InCubic }
                        }

                        color: dividerMouseArea.drag.active
                             ? DarkPalette.midlight
                             : (dividerMouseArea.containsMouse ? DarkPalette.mid : DarkPalette.middark)

                        Behavior on color {
                            ColorAnimation { duration: 90; easing.type: Easing.InCubic }
                        }
                    }

                    MouseArea {
                        id: dividerMouseArea
                        anchors.fill: parent
                        cursorShape: Qt.SplitHCursor
                        hoverEnabled: true

                        property bool dragging: false

                        onPressed: dragging = true
                        onReleased: dragging = false
                        onCanceled: dragging = false

                        onPositionChanged: {
                            if (!dragging)
                                return;
                            var dividerCentreX = dividerHandle.x + mouseX;
                            var clamped = Math.max(80, Math.min(fieldsArea.width - 80, dividerCentreX));
                            WorkspaceState.labelColumnWidth = Math.round(clamped);
                        }
                    }
                }
            }
        }
    }
}
