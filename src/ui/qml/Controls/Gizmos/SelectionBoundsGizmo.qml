import QtQuick
import QtQuick3D

import Pointcaster 1.0

// glue connecting a BoundsGizmo to a configuration adapter

Item {
    id: root

    required property View3D view3d
    required property var targetAdapter

    property matrix4x4 parentWorldTransform: Qt.matrix4x4()

    property string boundsPath: "transform/bounds"

    readonly property real limitMetres: 32.767
    readonly property real minExtentMetres: 0.05

    // Bounds as last read from (or written to) the config, in metres.
    property vector3d configMin: Qt.vector3d(0, 0, 0)
    property vector3d configMax: Qt.vector3d(0, 0, 0)

    readonly property bool dragging: gizmo.dragging

    anchors.fill: parent

    // ── Helpers ──

    function _toVector3d(value, fallback) {
        if (value === undefined || value === null || value.x === undefined)
            return fallback;
        return Qt.vector3d(value.x, value.y, value.z);
    }

    // parentWorldTransform carries its translation in metres
    function _sceneParentWorld(matrix) {
        return Qt.matrix4x4(matrix.m11, matrix.m12, matrix.m13, matrix.m14 * 100, matrix.m21, matrix.m22, matrix.m23, matrix.m24 * 100, matrix.m31, matrix.m32, matrix.m33, matrix.m34 * 100, matrix.m41, matrix.m42, matrix.m43, matrix.m44);
    }

    function _toScene(metres) {
        return Qt.vector3d(metres.x * 100, metres.y * 100, metres.z * 100);
    }

    function _toMetres(scene) {
        return Qt.vector3d(scene.x / 100, scene.y / 100, scene.z / 100);
    }

    // ── Config <-> gizmo ──

    function refreshFromConfig() {
        if (gizmo.dragging)
            return;

        if (!targetAdapter)
            return;

        // whatever the config carries, falling back on the same default the
        // numeric editor would offer rather than a second set of numbers here
        var bounds = targetAdapter.value(root.boundsPath);
        if (!bounds || bounds.min === undefined)
            bounds = targetAdapter.defaultValue(root.boundsPath);

        configMin = _toVector3d(bounds ? bounds.min : null, configMin);
        configMax = _toVector3d(bounds ? bounds.max : null, configMax);
    }

    function _writeBounds(minMetres, maxMetres, commit) {
        if (!root.targetAdapter)
            return;

        var payload = {
            min: minMetres,
            max: maxMetres
        };

        if (!commit) {
            root.targetAdapter.setPreview(root.boundsPath, payload);
            return;
        }

        root.targetAdapter.set(root.boundsPath, payload);

        root.configMin = payload.min;
        root.configMax = payload.max;
    }

    onTargetAdapterChanged: refreshFromConfig()

    onBoundsPathChanged: refreshFromConfig()

    Component.onCompleted: refreshFromConfig()

    Connections {
        target: root.targetAdapter
        ignoreUnknownSignals: true

        function onFieldChanged(path) {
            if (path === root.boundsPath)
                root.refreshFromConfig();
        }
    }

    BoundsGizmo {
        id: gizmo

        anchors.fill: parent

        view3d: root.view3d

        minPosition: root._toScene(root.configMin)
        maxPosition: root._toScene(root.configMax)
        parentWorld: root._sceneParentWorld(root.parentWorldTransform)

        limitLow: -root.limitMetres * 100
        limitHigh: root.limitMetres * 100
        minExtent: root.minExtentMetres * 100

        // 100mm, in scene units
        snapIncrement: 10

        boxColor: ThemeColors.yellow
        xAxisColor: ThemeColors.red
        yAxisColor: ThemeColors.green
        zAxisColor: ThemeColors.blue

        onBoundsPreview: (newMin, newMax) => root._writeBounds(root._toMetres(newMin), root._toMetres(newMax), false)

        onBoundsCommitted: (newMin, newMax) => root._writeBounds(root._toMetres(newMin), root._toMetres(newMax), true)
    }
}
