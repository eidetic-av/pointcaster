import QtQuick
import QtQuick.Layouts
import Pointcaster.Workspace 1.0

Row {
    id: root

    property real componentSpacing: 6
    spacing: componentSpacing

    property bool boundEnabled: true
    property var minValue: undefined
    property var maxValue: undefined
    property var defaultValue: undefined

    property var boundValue: Qt.vector3d(0, 0, 0)

    signal commitValue(var v3)

    readonly property real eachWidth: width > 0 ? Math.max(0, (width - 2 * componentSpacing) / 3) : -1

    function _componentFrom(v, axisIndex, fallback) {
        if (v === undefined || v === null)
            return fallback;
        if (v.x !== undefined) {
            if (axisIndex === 0)
                return v.x;
            if (axisIndex === 1)
                return v.y;
            return v.z;
        }

        if (Array.isArray(v) && v.length > axisIndex)
            return v[axisIndex];

        return fallback;
    }

    Repeater {
        model: [
            {
                label: "X",
                index: 0
            },
            {
                label: "Y",
                index: 1
            },
            {
                label: "Z",
                index: 2
            }
        ]

        delegate: Row {
            spacing: 4
            width: root.eachWidth > 0 ? root.eachWidth : implicitWidth

            Text {
                text: modelData.label
                font.pixelSize: 12
                color: DarkPalette.text
                verticalAlignment: Text.AlignVCenter
            }

            DragFloat {
                boundEnabled: root.boundEnabled
                width: parent.width - 12

                minValue: root._componentFrom(root.minValue, modelData.index, undefined)
                maxValue: root._componentFrom(root.maxValue, modelData.index, undefined)
                defaultValue: root._componentFrom(root.defaultValue, modelData.index, undefined)

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
