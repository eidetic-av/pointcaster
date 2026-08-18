import QtQuick

import Pointcaster 1.0

// the Editor for a position_bounds field -- an AABB
Item {
    id: root

    property font font: Scaling.uiFont
    property font axisFont: Scaling.uiSmallFont

    // per-component range, from what int16 millimetres can reach
    property real limit: 32.767

    property var minValue: -root.limit
    property var maxValue: root.limit
    property var defaultValue: undefined

    property var boundValue: ({
            min: Qt.vector3d(0, 0, 0),
            max: Qt.vector3d(0, 0, 0)
        })

    property int tagWidth: Math.round(26 * Scaling.uiScale)
    property int tagSpacing: Math.round(4 * Scaling.uiScale)

    signal commitValue(var bounds)

    implicitHeight: Math.round(60 * Scaling.uiScale)

    function _toVector3d(value) {
        if (!value || value.x === undefined)
            return Qt.vector3d(0, 0, 0);
        return Qt.vector3d(Number(value.x) || 0, Number(value.y) || 0, Number(value.z) || 0);
    }

    readonly property vector3d minPosition: root._toVector3d(root.boundValue ? root.boundValue.min : null)
    readonly property vector3d maxPosition: root._toVector3d(root.boundValue ? root.boundValue.max : null)

    readonly property real halfHeight: root.height / 2
    readonly property real fieldX: root.tagWidth + root.tagSpacing
    readonly property real fieldWidth: Math.max(0, root.width - root.fieldX)

    function syncFields() {
        minField.boundValue = root.minPosition;
        maxField.boundValue = root.maxPosition;
    }

    onBoundValueChanged: root.syncFields()
    Component.onCompleted: root.syncFields()

    // Keeps the box from inverting: a face pushed past its opposite stops there
    // rather than producing an empty volume that silently discards every point.
    function _commit(isMax, position) {
        var low = root.minPosition;
        var high = root.maxPosition;

        if (isMax) {
            high = Qt.vector3d(Math.max(position.x, low.x), Math.max(position.y, low.y), Math.max(position.z, low.z));
        } else {
            low = Qt.vector3d(Math.min(position.x, high.x), Math.min(position.y, high.y), Math.min(position.z, high.z));
        }

        root.boundValue = {
            min: low,
            max: high
        };
        root.commitValue(root.boundValue);
    }

    Item {
        id: minRow

        y: 0
        width: root.width
        height: root.halfHeight

        Text {
            width: root.tagWidth
            anchors.verticalCenter: parent.verticalCenter
            horizontalAlignment: Text.AlignRight
            text: qsTr("min")
            font: root.axisFont
            color: ThemeColors.readOnlyText
        }

        DragFloat3 {
            id: minField

            x: root.fieldX
            width: root.fieldWidth
            anchors.verticalCenter: parent.verticalCenter

            font: root.font
            axisFont: root.axisFont
            minValue: root.minValue
            maxValue: root.maxValue
            defaultValue: root.defaultValue ? root.defaultValue.min : undefined

            onCommitValue: position => root._commit(false, position)
        }
    }

    Item {
        id: maxRow

        y: root.halfHeight
        width: root.width
        height: root.halfHeight

        Text {
            width: root.tagWidth
            anchors.verticalCenter: parent.verticalCenter
            horizontalAlignment: Text.AlignRight
            text: qsTr("max")
            font: root.axisFont
            color: ThemeColors.readOnlyText
        }

        DragFloat3 {
            id: maxField

            x: root.fieldX
            width: root.fieldWidth
            anchors.verticalCenter: parent.verticalCenter

            font: root.font
            axisFont: root.axisFont
            minValue: root.minValue
            maxValue: root.maxValue
            defaultValue: root.defaultValue ? root.defaultValue.max : undefined

            onCommitValue: position => root._commit(true, position)
        }
    }
}
