import QtQuick
import QtQuick3D

import Gizmo3D

Item {
    id: root

    required property View3D view3d
    required property Node targetNode
    required property var targetAdapter

    property real size: 80

    property vector3d dragStartPos: Qt.vector3d(0, 0, 0)
    property quaternion dragStartRot: Qt.quaternion(1, 0, 0, 0)
    property vector3d dragStartEuler: Qt.vector3d(0, 0, 0)
    property vector3d dragStartScale: Qt.vector3d(1, 1, 1)

    anchors.fill: parent

    GlobalGizmo {
        id: gizmo
        view3d: root.view3d
        targetNode: root.targetNode
        mode: GizmoEnums.Mode.All
        gizmoSize: root.size
        anchors.fill: parent
    }

    // Translation signal connections
    Connections {
        target: gizmo
        ignoreUnknownSignals: true

        function onAxisTranslationStarted(axis) {
            root.dragStartPos = root.targetNode.position;
        }

        function onAxisTranslationDelta(axis, transformMode, delta, snapActive) {
            // Convert axis number to 3D direction based on transform mode
            var axisDirection;
            if (transformMode === GizmoEnums.TransformMode.Local) {
                // Calculate local axes from target node's scene rotation (includes parent transforms)
                var localAxes = GizmoMath.getLocalAxes(root.targetNode.sceneRotation);
                axisDirection = axis === GizmoEnums.Axis.X ? localAxes.x : axis === GizmoEnums.Axis.Y ? localAxes.y : localAxes.z;
            } else {
                // World mode: use global X/Y/Z axes
                axisDirection = axis === GizmoEnums.Axis.X ? Qt.vector3d(1, 0, 0) : axis === GizmoEnums.Axis.Y ? Qt.vector3d(0, 1, 0) : Qt.vector3d(0, 0, 1);
            }

            // Apply translation along the axis direction
            var deltaVec = Qt.vector3d(axisDirection.x * delta, axisDirection.y * delta, axisDirection.z * delta);
            var newQtPos = Qt.vector3d(root.dragStartPos.x + deltaVec.x, root.dragStartPos.y + deltaVec.y, root.dragStartPos.z + deltaVec.z);
            root.targetNode.position = newQtPos;
            if (root.targetAdapter) {
                var unscaledPos = Qt.vector3d(newQtPos.x * 0.01, newQtPos.y * 0.01, newQtPos.z * 0.01);
                root.targetAdapter.set("transform/position", unscaledPos);
            }
        }

        function onPlaneTranslationStarted(plane) {
            root.dragStartPos = root.targetNode.position;
        }

        function onPlaneTranslationDelta(plane, transformMode, delta, snapActive) {
            var deltaVec;
            if (transformMode === GizmoEnums.TransformMode.Local) {
                // Local mode: delta components are along local axes, convert to world space
                var localAxes = GizmoMath.getLocalAxes(root.targetNode.sceneRotation);
                deltaVec = Qt.vector3d(localAxes.x.x * delta.x + localAxes.y.x * delta.y + localAxes.z.x * delta.z, localAxes.x.y * delta.x + localAxes.y.y * delta.y + localAxes.z.y * delta.z, localAxes.x.z * delta.x + localAxes.y.z * delta.y + localAxes.z.z * delta.z);
            } else {
                // World mode: delta is already in world space
                deltaVec = delta;
            }
            var newQtPos = Qt.vector3d(root.dragStartPos.x + deltaVec.x, root.dragStartPos.y + deltaVec.y, root.dragStartPos.z + deltaVec.z);
            root.targetNode.position = newQtPos;
            if (root.targetAdapter) {
                var unscaledPos = Qt.vector3d(newQtPos.x * 0.01, newQtPos.y * 0.01, newQtPos.z * 0.01);
                root.targetAdapter.set("transform/position", unscaledPos);
            }
        }
    }

    // Rotation signal connections
    Connections {
        target: gizmo
        ignoreUnknownSignals: true

        function onRotationStarted(axis) {
            root.dragStartEuler = root.targetAdapter ? root.targetAdapter.value("transform/rotation") : root.targetNode.eulerRotation;
        }

        function onRotationDelta(axis, transformMode, angleDegrees, snapActive) {
            const sign = axis === GizmoEnums.Axis.X ? 1 : axis === GizmoEnums.Axis.Y ? 1 : 1;

            const delta = angleDegrees * sign;
            const e = root.dragStartEuler;

            let newEuler;
            if (axis === GizmoEnums.Axis.X)
                newEuler = Qt.vector3d(e.x + delta, e.y, e.z);
            else if (axis === GizmoEnums.Axis.Y)
                newEuler = Qt.vector3d(e.x, e.y + delta, e.z);
            else
                newEuler = Qt.vector3d(e.x, e.y, e.z + delta);

            root.targetNode.eulerRotation = newEuler;
            if (root.targetAdapter) {
                // root.targetAdapter.set("transform/rotation", Qt.vector3d(newEuler.x, newEuler.z, newEuler.y));
                root.targetAdapter.set("transform/rotation", Qt.vector3d(newEuler.x, newEuler.y, newEuler.z));
            }
        }
    }

    // Scale signal connections
    Connections {
        target: gizmo
        ignoreUnknownSignals: true

        function onScaleStarted(axis) {
            root.dragStartScale = root.targetAdapter ? root.targetAdapter.value("transform/scale") : root.targetNode.scale;
        }

        function onScaleDelta(axis, transformMode, scaleFactor, snapActive) {
            // Scale is axis-aligned regardless of transform mode
            var newScale = root.targetNode.scale;
            if (axis === GizmoEnums.Axis.Uniform) {
                // Uniform scaling
                newScale = Qt.vector3d(root.dragStartScale.x * scaleFactor, root.dragStartScale.y * scaleFactor, root.dragStartScale.z * scaleFactor);
            } else {
                // Axis-constrained scaling
                if (axis === GizmoEnums.Axis.X) {
                    newScale = Qt.vector3d(root.dragStartScale.x * scaleFactor, root.dragStartScale.y, root.dragStartScale.z);
                } else if (axis === GizmoEnums.Axis.Y) {
                    newScale = Qt.vector3d(root.dragStartScale.x, root.dragStartScale.y * scaleFactor, root.dragStartScale.z);
                } else if (axis === GizmoEnums.Axis.Z) {
                    newScale = Qt.vector3d(root.dragStartScale.x, root.dragStartScale.y, root.dragStartScale.z * scaleFactor);
                }
            }
            if (root.targetAdapter) {
                root.targetAdapter.set("transform/scale", newScale);
            }
        }
    }
}
