import QtQuick
import QtQuick.Layouts
import QtQuick.Shapes

import Pointcaster 1.0

Row {
    id: root
    spacing: 0

    Layout.minimumWidth: Math.round(260 * Scaling.uiScale)

    property font font: Scaling.uiFont
    property font axisFont: Scaling.axisLabelFont

    property real componentSpacing: Math.round(6 * Scaling.uiScale)

    property var minValue: undefined
    property var maxValue: undefined
    property var defaultValue: undefined

    property var boundValue: Qt.vector3d(0, 0, 0)

    property int labelWidth: Math.round(14 * Scaling.uiScale)
    property int labelHeight: Math.round(16 * Scaling.uiScale)
    property int labelLeftRadius: Math.round(3 * Scaling.uiScale)
    property color labelBackgroundColor: ThemeColors.almostdark

    signal commitValue(var value)
    // mid-drag value, live but not yet on the undo stack
    signal previewValue(var value)

    function _axisValue(vector, axisIndex) {
        if (axisIndex === 0)
            return Number(vector.x) || 0;
        if (axisIndex === 1)
            return Number(vector.y) || 0;
        return Number(vector.z) || 0;
    }

    function _axisDefault(axisIndex) {
        const defaultValue = root.defaultValue;
        if (defaultValue === undefined || defaultValue === null)
            return undefined;
        if (typeof defaultValue === "object")
            return (defaultValue.x === undefined) ? undefined : root._axisValue(defaultValue, axisIndex);
        return defaultValue; // a scalar default applies to every axis
    }

    // returns the vector an axis edit produces, or null if nothing moved
    function _movedVector(axisIndex, componentValue) {
        const next = root._withAxis(root.boundValue, axisIndex, componentValue);
        if (next.x === root.boundValue.x && next.y === root.boundValue.y && next.z === root.boundValue.z)
            return null;
        return next;
    }

    function _withAxis(vector, axisIndex, newComponentValue) {
        const x = (axisIndex === 0) ? newComponentValue : (Number(vector.x) || 0);
        const y = (axisIndex === 1) ? newComponentValue : (Number(vector.y) || 0);
        const z = (axisIndex === 2) ? newComponentValue : (Number(vector.z) || 0);
        return Qt.vector3d(x, y, z);
    }

    Item {
        id: leftPadding
        width: Math.round(componentSpacing / 2)
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
                    color: ThemeColors.red
                },
                {
                    label: "Y",
                    index: 1,
                    color: ThemeColors.green
                },
                {
                    label: "Z",
                    index: 2,
                    color: ThemeColors.blue
                }
            ]

            delegate: Row {
                id: axisRow
                spacing: 0
                width: contentRow.eachWidth > 0 ? contentRow.eachWidth : implicitWidth

                readonly property int axisIndex: modelData.index
                readonly property var axisDefault: root._axisDefault(axisIndex)

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
                        font: root.axisFont
                        color: modelData.color
                    }
                }

                DragFloat {
                    id: dragFloat
                    width: parent.width - axisLabel.width
                    font: root.font

                    backgroundRadius: root.labelLeftRadius
                    backgroundLeftRadius: 0

                    // a vector component is dragged rather than clicked, and at
                    // these widths the arrows are a two pixel target that would
                    // cost the number 12px of padding to show
                    showArrowButtons: false

                    minValue: root.minValue
                    maxValue: root.maxValue
                    defaultValue: axisRow.axisDefault

                    boundValue: 0.0

                    onPreviewValue: function (componentValue) {
                        const next = root._movedVector(axisIndex, componentValue);
                        if (!next)
                            return;

                        root.boundValue = next;
                        root.previewValue(next);
                    }

                    // the previews during a drag have already carried
                    // boundValue to where this lands, so a commit matching it
                    // is the normal case -- it still has to be emitted to close
                    // the gesture and put it on the undo stack
                    onCommitValue: function (componentValue) {
                        root.boundValue = root._withAxis(root.boundValue, axisIndex, componentValue);
                        root.commitValue(root.boundValue);
                    }
                }
            }
        }
    }
}
