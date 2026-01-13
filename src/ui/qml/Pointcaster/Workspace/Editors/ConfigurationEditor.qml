import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

Item {
    id: root
    property var model: null

    property string emptyTitle: "Properties"

    // Max scroll height for the list
    property int maxBodyHeight: 320

    // Shared width of the left "label" column (draggable divider)
    property int labelColumnWidth: Math.round(width * 0.35)

    // Collapsible group state
    property bool expanded: true

    // ---- Layout ----
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
                color: DarkPalette.text
            }

            MouseArea {
                id: headerMouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: root.expanded = !root.expanded
            }
        }

        // body (properties list)
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

            // empty state when no selection
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

                // Property rows
                Column {
                    id: propertiesColumn
                    width: propertiesFlick.width
                    spacing: 4

                    Repeater {
                        model: root.model ? root.model.fieldCount() : 0

                        delegate: Item {
                            required property int index

                            width: propertiesColumn.width
                            height: Math.max(nameText.implicitHeight, valueField.implicitHeight) + 6

                            visible: root.model && !root.model.isHidden(index)

                            // Label region (left column)
                            Item {
                                id: labelContainer
                                anchors {
                                    left: parent.left
                                    top: parent.top
                                    bottom: parent.bottom
                                }
                                width: root.labelColumnWidth

                                Text {
                                    id: nameText
                                    anchors {
                                        left: parent.left
                                        right: parent.right
                                        verticalCenter: parent.verticalCenter
                                    }
                                    horizontalAlignment: Text.AlignLeft
                                    color: DarkPalette.text
                                    text: root.model ? root.model.fieldName(index) : ""
                                }
                            }

                            // value region (right column)
                            TextField {
                                id: valueField
                                anchors {
                                    left: labelContainer.right
                                    right: parent.right
                                    verticalCenter: parent.verticalCenter
                                }

                                text: root.model ? root.model.fieldValue(index) : ""
                                enabled: root.model ? !root.model.isDisabled(index) : false

                                onEditingFinished: {
                                    if (!root.model)
                                        return;
                                    root.model.setFieldValue(index, text);
                                }
                            }
                        }
                    }
                }

                // vertical divider handle (two-layer)
                Rectangle {
                    id: dividerHandle
                    x: root.labelColumnWidth - width / 2
                    width: 4
                    anchors {
                        top: parent.top
                        bottom: parent.bottom
                        bottomMargin: 1
                    }
                    color: DarkPalette.base

                    Rectangle {
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: 1
                        height: parent.height
                        color: DarkPalette.mid
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.SplitHCursor

                        drag.target: dividerHandle
                        drag.axis: Drag.XAxis
                        drag.minimumX: 80
                        drag.maximumX: root.width - 80

                        onPositionChanged: {
                            root.labelColumnWidth = dividerHandle.x + dividerHandle.width / 2;
                        }
                    }
                }
            }
        }
    }
}
