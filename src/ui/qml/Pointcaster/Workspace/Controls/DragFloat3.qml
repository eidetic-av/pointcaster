import QtQuick
import QtQuick.Layouts
import QtQuick.Shapes

import Pointcaster.Workspace 1.0

Row {
    id: root
    spacing: 0

    Layout.minimumWidth: 260

    property real componentSpacing: 6

    property var minValue: undefined
    property var maxValue: undefined
    property var defaultValue: undefined

    property var boundValue: Qt.vector3d(0, 0, 0)

    property int labelWidth: 16
    property int labelHeight: 16
    property int labelLeftRadius: 3
    property color labelBackgroundColor: DarkPalette.almostdark

    signal commitValue(var v3)

    Item {
        id: leftPadding
        width: 3
        height: 1
    }

    Row {
        id: contentRow
        spacing: root.componentSpacing

        width: Math.max(0, root.width - leftPadding.width)

        readonly property real eachWidth: width > 0 ? Math.max(0, (width - 2 * root.componentSpacing) / 3) : -1

        Repeater {
            model: [
                {
                    label: "X",
                    index: 0,
                    color: DarkPalette.red
                },
                {
                    label: "Y",
                    index: 1,
                    color: DarkPalette.green
                },
                {
                    label: "Z",
                    index: 2,
                    color: DarkPalette.blue
                }
            ]

            delegate: Row {
                spacing: 0
                width: contentRow.eachWidth > 0 ? contentRow.eachWidth : implicitWidth

                Item {
                    id: axisLabel
                    width: root.labelWidth
                    height: Math.max(root.labelHeight, dragFloat.implicitHeight, dragFloat.height)

                    // rectangle background for the label has the
                    // left side rounded only (top-left + bottom-left)
                    Shape {
                        anchors.fill: parent
                        antialiasing: true

                        ShapePath {
                            strokeWidth: 0
                            fillColor: root.labelBackgroundColor

                            startX: 0
                            startY: root.labelLeftRadius

                            // top-left corner arc right
                            PathArc {
                                x: root.labelLeftRadius
                                y: 0
                                radiusX: root.labelLeftRadius
                                radiusY: root.labelLeftRadius
                            }

                            // top edge to top-right (square)
                            PathLine {
                                x: axisLabel.width
                                y: 0
                            }

                            // right edge down (square)
                            PathLine {
                                x: axisLabel.width
                                y: axisLabel.height
                            }

                            // bottom edge to before bottom-left arc
                            PathLine {
                                x: root.labelLeftRadius
                                y: axisLabel.height
                            }

                            // bottom-left corner arc up
                            PathArc {
                                x: 0
                                y: axisLabel.height - root.labelLeftRadius
                                radiusX: root.labelLeftRadius
                                radiusY: root.labelLeftRadius
                            }

                            // left edge back to start
                            PathLine {
                                x: 0
                                y: root.labelLeftRadius
                            }
                        }
                    }

                    Text {
                        anchors.centerIn: parent
                        text: modelData.label
                        font.pixelSize: 11
                        font.weight: Font.Bold
                        color: modelData.color
                    }
                }

                DragFloat {
                    id: dragFloat
                    width: parent.width - axisLabel.width

                    minValue: root.minValue
                    maxValue: root.maxValue
                    defaultValue: root.defaultValue

                    boundValue: modelData.index === 0 ? root.boundValue.x : modelData.index === 1 ? root.boundValue.y : root.boundValue.z

                    onCommitValue: {
                        if (modelData.index === 0)
                            root.boundValue = Qt.vector3d(boundValue, root.boundValue.y, root.boundValue.z);
                        else if (modelData.index === 1)
                            root.boundValue = Qt.vector3d(root.boundValue.x, boundValue, root.boundValue.z);
                        else
                            root.boundValue = Qt.vector3d(root.boundValue.x, root.boundValue.y, boundValue);

                        root.commitValue(root.boundValue);
                    }
                }
            }
        }
    }
}
