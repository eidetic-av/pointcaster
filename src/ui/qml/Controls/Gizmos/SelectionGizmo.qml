import QtQuick
import QtQuick3D

import Gizmo3D

Item {
    id: root

    required property View3D view3d
    required property Node targetNode
    required property var targetAdapter
    required property bool cameraTarget

    required property int mode

    property string cameraPositionPath: "camera/position"

    property matrix4x4 parentWorldTransform: Qt.matrix4x4()

    property real size: 80

    // ── Live drag state (read by SessionView for immediate visual feedback) ──
    property bool dragging: false
    property vector3d dragPosition: Qt.vector3d(0, 0, 0)
    property vector3d dragRotation: Qt.vector3d(0, 0, 0)
    property vector3d dragScale: Qt.vector3d(1, 1, 1)

    // ── Private drag-start snapshots ──
    property vector3d _startPos: Qt.vector3d(0, 0, 0)
    property vector3d _startEuler: Qt.vector3d(0, 0, 0)
    property vector3d _startScale: Qt.vector3d(1, 1, 1)

    anchors.fill: parent

    GlobalGizmo {
        id: gizmo
        view3d: root.view3d
        targetNode: root.targetNode
        mode: root.mode
        gizmoSize: root.size
        anchors.fill: parent
    }

    // ── Helpers ──

    function _vec3(val, fallback) {
        if (val === undefined || val === null || val.x === undefined)
            return fallback;
        return Qt.vector3d(val.x, val.y, val.z);
    }

    function snapshotCurrent() {
        dragPosition = targetNode.position;

        if (cameraTarget) {
            // Camera/operator targets only carry a position; take orientation
            // and scale from the node rather than reading absent transform/* paths.
            dragRotation = targetNode.eulerRotation;
            dragScale = targetNode.scale;
        } else {
            dragRotation = targetAdapter ? _vec3(targetAdapter.value("transform/rotation"), targetNode.eulerRotation) : targetNode.eulerRotation;
            dragScale = targetAdapter ? _vec3(targetAdapter.value("transform/scale"), targetNode.scale) : targetNode.scale;
        }

        _startPos = dragPosition;
        _startEuler = dragRotation;
        _startScale = dragScale;
    }

    function commitTransform() {
        if (!dragging)
            return;
        if (!targetAdapter) {
            dragging = false;
            return;
        }

        if (cameraTarget) {
            targetAdapter.set(root.cameraPositionPath, Qt.vector3d(dragPosition.x * 0.01, dragPosition.y * 0.01, dragPosition.z * 0.01));
        } else {
            // dragPosition is world scene units, convert to world metres, then strip
            // the parent transform to get the device/group local position to store.
            var worldM = Qt.vector3d(dragPosition.x * 0.01, dragPosition.y * 0.01, dragPosition.z * 0.01);
            var localM = root.parentWorldTransform.inverted().times(worldM);
            targetAdapter.set("transform/position", localM);
            targetAdapter.set("transform/rotation", dragRotation);
            targetAdapter.set("transform/scale", dragScale);
        }

        dragging = false;
    }

    // ── Translation ──

    Connections {
        target: gizmo
        ignoreUnknownSignals: true

        function onAxisTranslationStarted(axis) {
            root.snapshotCurrent();
            root.dragging = true;
        }

        function onAxisTranslationDelta(axis, transformMode, delta, snapActive) {
            var dir;
            if (transformMode === GizmoEnums.TransformMode.Local) {
                var local = GizmoMath.getLocalAxes(root.targetNode.sceneRotation);
                dir = axis === GizmoEnums.Axis.X ? local.x : axis === GizmoEnums.Axis.Y ? local.y : local.z;
            } else {
                dir = axis === GizmoEnums.Axis.X ? Qt.vector3d(1, 0, 0) : axis === GizmoEnums.Axis.Y ? Qt.vector3d(0, 1, 0) : Qt.vector3d(0, 0, 1);
            }
            root.dragPosition = Qt.vector3d(root._startPos.x + dir.x * delta, root._startPos.y + dir.y * delta, root._startPos.z + dir.z * delta);
        }

        function onAxisTranslationEnded(axis) {
            root.commitTransform();
        }

        function onPlaneTranslationStarted(plane) {
            root.snapshotCurrent();
            root.dragging = true;
        }

        function onPlaneTranslationDelta(plane, transformMode, delta, snapActive) {
            var d;
            if (transformMode === GizmoEnums.TransformMode.Local) {
                var local = GizmoMath.getLocalAxes(root.targetNode.sceneRotation);
                d = Qt.vector3d(local.x.x * delta.x + local.y.x * delta.y + local.z.x * delta.z, local.x.y * delta.x + local.y.y * delta.y + local.z.y * delta.z, local.x.z * delta.x + local.y.z * delta.y + local.z.z * delta.z);
            } else {
                d = delta;
            }
            root.dragPosition = Qt.vector3d(root._startPos.x + d.x, root._startPos.y + d.y, root._startPos.z + d.z);
        }

        function onPlaneTranslationEnded(plane) {
            root.commitTransform();
        }
    }

    // ── Rotation ──

    Connections {
        target: gizmo
        ignoreUnknownSignals: true

        function onRotationStarted(axis) {
            root.snapshotCurrent();
            root.dragging = true;
        }

        function onRotationDelta(axis, transformMode, angleDegrees, snapActive) {
            var e = root._startEuler;
            if (axis === GizmoEnums.Axis.X)
                root.dragRotation = Qt.vector3d(e.x + angleDegrees, e.y, e.z);
            else if (axis === GizmoEnums.Axis.Y)
                root.dragRotation = Qt.vector3d(e.x, e.y + angleDegrees, e.z);
            else
                root.dragRotation = Qt.vector3d(e.x, e.y, e.z + angleDegrees);
        }

        function onRotationEnded(axis) {
            root.commitTransform();
        }
    }

    // ── Scale ──

    Connections {
        target: gizmo
        ignoreUnknownSignals: true

        function onScaleStarted(axis) {
            root.snapshotCurrent();
            root.dragging = true;
        }

        function onScaleDelta(axis, transformMode, scaleFactor, snapActive) {
            var s = root._startScale;
            if (axis === GizmoEnums.Axis.Uniform)
                root.dragScale = Qt.vector3d(s.x * scaleFactor, s.y * scaleFactor, s.z * scaleFactor);
            else if (axis === GizmoEnums.Axis.X)
                root.dragScale = Qt.vector3d(s.x * scaleFactor, s.y, s.z);
            else if (axis === GizmoEnums.Axis.Y)
                root.dragScale = Qt.vector3d(s.x, s.y * scaleFactor, s.z);
            else
                root.dragScale = Qt.vector3d(s.x, s.y, s.z * scaleFactor);
        }

        function onScaleEnded(axis) {
            root.commitTransform();
        }
    }

    // ── Fallback: if gizmo doesn't emit *Ended signals ──

    Connections {
        target: gizmo
        ignoreUnknownSignals: true
        function onActiveChanged() {
            if (!gizmo.active && root.dragging)
                root.commitTransform();
        }
    }
}
