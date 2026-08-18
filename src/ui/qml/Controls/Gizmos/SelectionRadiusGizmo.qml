import QtQuick
import QtQuick3D

import Pointcaster 1.0

// glue connecting a RadiusGizmo to a configuration adapter

Item {
    id: root

    required property View3D view3d
    required property var targetAdapter

    property matrix4x4 parentWorldTransform: Qt.matrix4x4()

    property string radiusPath: ""

    readonly property real limitMetres: 65.535

    // the distance as last read from (or written to) the config, in metres
    property real configRadius: 0

    readonly property bool dragging: gizmo.dragging

    anchors.fill: parent

    // ── Helpers ──

    // parentWorldTransform carries its translation in metres
    function _sceneParentWorld(matrix) {
        return Qt.matrix4x4(matrix.m11, matrix.m12, matrix.m13, matrix.m14 * 100, matrix.m21, matrix.m22, matrix.m23, matrix.m24 * 100, matrix.m31, matrix.m32, matrix.m33, matrix.m34 * 100, matrix.m41, matrix.m42, matrix.m43, matrix.m44);
    }

    // ── Config <-> gizmo ──

    // a distance reads as a bare number when it carries no switch, and as a
    // map holding one under "value" when it does
    function _radiusOf(value) {
        if (value === undefined || value === null)
            return undefined;
        const number = Number(value.value !== undefined ? value.value : value);
        return isNaN(number) ? undefined : number;
    }

    function refreshFromConfig() {
        if (gizmo.dragging)
            return;

        if (!targetAdapter || !radiusPath)
            return;

        var held = _radiusOf(targetAdapter.value(root.radiusPath));
        if (held === undefined)
            held = _radiusOf(targetAdapter.defaultValue(root.radiusPath));

        if (held !== undefined)
            configRadius = held;
    }

    function _writeRadius(metres, commit) {
        if (!root.targetAdapter || !root.radiusPath)
            return;

        // only the distance goes out, so whatever switch the field carries is
        // left exactly as the user set it
        var payload = {
            value: metres
        };

        if (!commit) {
            root.targetAdapter.setPreview(root.radiusPath, payload);
            return;
        }

        root.targetAdapter.set(root.radiusPath, payload);
        root.configRadius = metres;
    }

    onTargetAdapterChanged: refreshFromConfig()

    onRadiusPathChanged: refreshFromConfig()

    Component.onCompleted: refreshFromConfig()

    Connections {
        target: root.targetAdapter
        ignoreUnknownSignals: true

        function onFieldChanged(path) {
            if (path === root.radiusPath)
                root.refreshFromConfig();
        }
    }

    RadiusGizmo {
        id: gizmo

        anchors.fill: parent

        view3d: root.view3d

        radius: root.configRadius * 100
        parentWorld: root._sceneParentWorld(root.parentWorldTransform)

        limitLow: 0
        limitHigh: root.limitMetres * 100

        // 100mm, in scene units
        snapIncrement: 10

        ringColor: ThemeColors.yellow

        onRadiusPreview: newRadius => root._writeRadius(newRadius / 100, false)

        onRadiusCommitted: newRadius => root._writeRadius(newRadius / 100, true)
    }
}
