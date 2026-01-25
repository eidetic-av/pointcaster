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

    function _axisValue(v3, axisIndex) {
        if (axisIndex === 0)
            return Number(v3.x) || 0;
        if (axisIndex === 1)
            return Number(v3.y) || 0;
        return Number(v3.z) || 0;
    }

    function _withAxis(v3, axisIndex, newComponentValue) {
        const x = (axisIndex === 0) ? newComponentValue : (Number(v3.x) || 0);
        const y = (axisIndex === 1) ? newComponentValue : (Number(v3.y) || 0);
        const z = (axisIndex === 2) ? newComponentValue : (Number(v3.z) || 0);
        return Qt.vector3d(x, y, z);
    }

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
                id: axisRow
                spacing: 0
                width: contentRow.eachWidth > 0 ? contentRow.eachWidth : implicitWidth

                readonly property int axisIndex: modelData.index

                function syncFromRoot() {
                    dragFloat.boundValue = root._axisValue(root.boundValue, axisIndex);
                }

                Component.onCompleted: syncFromRoot()

                Connections {
                    target: root
                    function onBoundValueChanged() {
                        axisRow.syncFromRoot();
                    }
                }

                Item {
                    id: axisLabel
                    width: root.labelWidth
                    height: Math.max(root.labelHeight, dragFloat.implicitHeight, dragFloat.height)

                    Shape {
                        anchors.fill: parent
                        antialiasing: true

                        ShapePath {
                            strokeWidth: 0
                            fillColor: root.labelBackgroundColor

                            startX: 0
                            startY: root.labelLeftRadius

                            PathArc {
                                x: root.labelLeftRadius
                                y: 0
                                radiusX: root.labelLeftRadius
                                radiusY: root.labelLeftRadius
                            }

                            PathLine {
                                x: axisLabel.width
                                y: 0
                            }
                            PathLine {
                                x: axisLabel.width
                                y: axisLabel.height
                            }
                            PathLine {
                                x: root.labelLeftRadius
                                y: axisLabel.height
                            }

                            PathArc {
                                x: 0
                                y: axisLabel.height - root.labelLeftRadius
                                radiusX: root.labelLeftRadius
                                radiusY: root.labelLeftRadius
                            }

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

                    boundValue: 0.0

                    onCommitValue: function (v) {
                        const next = root._withAxis(root.boundValue, axisIndex, v);
                        if (next.x === root.boundValue.x && next.y === root.boundValue.y && next.z === root.boundValue.z)
                            return;

                        root.boundValue = next;
                        root.commitValue(next);
                    }
                }
            }
        }
    }
}
