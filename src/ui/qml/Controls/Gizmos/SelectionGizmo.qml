import QtQuick
import QtQuick3D

import Pointcaster 1.0

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

    readonly property quaternion parentRotation: TransformUtils.rotationFromMatrix(parentWorldTransform)

    property real size: 80

    readonly property bool gizmoActive: gizmo.isActive

    // ── Live drag state (read by SessionView for immediate visual feedback) ──
    property bool dragging: false
    property vector3d dragPosition: Qt.vector3d(0, 0, 0)
    property quaternion dragOrientation: Qt.quaternion(1, 0, 0, 0)

    // ── Private drag-start snapshots ──
    property vector3d _startPos: Qt.vector3d(0, 0, 0)
    property quaternion _startOrientation: Qt.quaternion(1, 0, 0, 0)
    property bool _rotated: false

    anchors.fill: parent

    // Translate and rotate only — transform/scale is edited numerically in the
    // configuration editor, never through the gizmo.
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
            dragOrientation = targetNode.rotation;
        } else {
            var localEuler = targetAdapter ? _vec3(targetAdapter.value("transform/rotation"), Qt.vector3d(0, 0, 0)) : Qt.vector3d(0, 0, 0);
            dragOrientation = root.parentRotation.times(TransformUtils.quaternionFromEuler(localEuler));
        }

        _startPos = dragPosition;
        _startOrientation = dragOrientation;
        _rotated = false;
    }

    function previewTransform() {
        if (!dragging || !targetAdapter)
            return;
        if (cameraTarget)
            return;

        var worldM = Qt.vector3d(dragPosition.x * 0.01, dragPosition.y * 0.01, dragPosition.z * 0.01);
        var localM = root.parentWorldTransform.inverted().times(worldM);
        targetAdapter.setPreview("transform/position", localM);

        if (_rotated) {
            var localQ = root.parentRotation.conjugated().times(dragOrientation);
            targetAdapter.setPreview("transform/rotation", TransformUtils.eulerFromQuaternion(localQ));
        }
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
            var worldM = Qt.vector3d(dragPosition.x * 0.01, dragPosition.y * 0.01, dragPosition.z * 0.01);
            var localM = root.parentWorldTransform.inverted().times(worldM);
            targetAdapter.set("transform/position", localM);
            if (_rotated) {
                var localQ = root.parentRotation.conjugated().times(dragOrientation);
                targetAdapter.set("transform/rotation", TransformUtils.eulerFromQuaternion(localQ));
            }
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
            root.previewTransform();
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
            root.previewTransform();
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
            var dir = axis === GizmoEnums.Axis.X ? Qt.vector3d(1, 0, 0) : axis === GizmoEnums.Axis.Y ? Qt.vector3d(0, 1, 0) : Qt.vector3d(0, 0, 1);
            var delta = TransformUtils.quaternionFromAxisAngle(dir, angleDegrees);
            root.dragOrientation = delta.times(root._startOrientation);
            root._rotated = true;
            root.previewTransform();
        }

        function onRotationEnded(axis) {
            root.commitTransform();
        }
    }

    // ── Fallback: if a gizmo doesn't emit its *Ended signal ──

    onGizmoActiveChanged: {
        if (!gizmoActive && dragging)
            commitTransform();
    }
}
