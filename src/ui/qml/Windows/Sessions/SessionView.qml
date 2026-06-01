import QtQuick
import QtQuick.Controls
import QtQuick3D
import QtQuick3D.Helpers

import Pointcaster 1.0

import Gizmo3D

Item {
    id: root

    required property var workspace
    required property var sessionAdapter
    required property var deviceAdapters
    property var cameraAdapter: null

    property var selectedDeviceAdapter: (workspace && deviceAdapters) ? deviceAdapters.length > 0 ? root.deviceAdapters[root.workspace.selectedDeviceIndex] : null : null
    property var selectedOperatorAdapter: workspace ? workspace.selectedOperatorAdapter ? workspace.selectedOperatorAdapter.configAdapter : null : null

    property var selectionPosition: selectionPositionOrDefault()
    function selectionPositionOrDefault() {
        var pos_mm = Qt.vector3d(0, 0, 0);
        if (selectedOperatorAdapter) {
            pos_mm = selectedOperatorAdapter.value("camera/position");
        } else if (selectedDeviceAdapter) {
            pos_mm = selectedDeviceAdapter.value("transform/position");
        }
        return Qt.vector3d(pos_mm.x * 100, pos_mm.y * 100, pos_mm.z * 100);
    }

    property var selectionScale: selectionScaleOrDefault()
    function selectionScaleOrDefault() {
        var scale = Qt.vector3d(1, 1, 1);
        if (selectedDeviceAdapter) {
            scale = selectedDeviceAdapter.value("transform/scale");
        }
        return scale;
    }

    property vector3d selectionRotation: selectionRotationOrDefault()
    function selectionRotationOrDefault() {
        var euler = Qt.vector3d(0, 0, 0);
        if (selectedDeviceAdapter) {
            euler = selectedDeviceAdapter.value("transform/rotation");
        }
        return euler;
    }

    signal selectionTransformUpdate

    onSelectionTransformUpdate: {
        selectionPosition = selectionPositionOrDefault();
        selectionScale = selectionScaleOrDefault();
        selectionRotation = selectionRotationOrDefault();
    }

    Connections {
        target: workspace
        function onSelectedDeviceIndexChanged() {
            root.selectionTransformUpdate();
        }
        function onSelectedOperatorAdapterChanged() {
            root.selectionTransformUpdate();
        }
    }

    readonly property real defaultCameraDistance: 250
    readonly property vector3d defaultOrbitOriginPosition: Qt.vector3d(0, 0, 0)
    readonly property quaternion defaultOrbitOriginRotation: {
        const pitch = -17;
        const yaw = 0;
        const pitchRad = pitch * Math.PI / 180;
        const yawRad = yaw * Math.PI / 180;
        const qPitch = Qt.quaternion(Math.cos(pitchRad * 0.5), Math.sin(pitchRad * 0.5), 0, 0);
        const qYaw = Qt.quaternion(Math.cos(yawRad * 0.5), 0, Math.sin(yawRad * 0.5), 0);
        return qYaw.times(qPitch);
    }

    property bool showBorder: false
    property color borderColor: ThemeColors.highlight

    // guards to prevent config<->UI updates from immediately re-committing (causing undo/redo feedback loops)
    property bool _applyingConfigCameraTransform: false
    property bool _applyingConfigCameraToggles: false
    property bool _committing_config: false

    function _vector3dFromAdapterPosition(p) {
        // p is expected to be QVector3D-ish in QML (has x/y/z)
        if (!p)
            return Qt.vector3d(0, 0, 0);

        return Qt.vector3d(Number(p.x) || 0, Number(p.y) || 0, Number(p.z) || 0);
    }

    function _quaternionFromAdapterRotation(r) {
        if (!r)
            return Qt.quaternion(1, 0, 0, 0);

        return Qt.quaternion(Number(r.scalar) || 1, Number(r.x) || 0, Number(r.y) || 0, Number(r.z) || 0);
    }

    function applyCameraTogglesFromConfig() {
        if (!root.cameraAdapter)
            return;

        root._applyingConfigCameraToggles = true;
        sessionControls.viewLocked = !!root.cameraAdapter.locked;
        sessionControls.gridEnabled = !!root.cameraAdapter.show_grid;
        sessionControls.orthographicEnabled = !!root.cameraAdapter.orthographic;
        // drop the guard next tick so any bindings/animations settle first
        Qt.callLater(function () {
            root._applyingConfigCameraToggles = false;
        });
    }

    function applyCameraTransformFromConfig() {
        if (!root.cameraAdapter)
            return;

        root._applyingConfigCameraTransform = true;
        orbitOrigin.position = _vector3dFromAdapterPosition(root.cameraAdapter.position);
        orbitOrigin.rotation = _quaternionFromAdapterRotation(root.cameraAdapter.rotation);
        camera.z = root.cameraAdapter.distance;
        // drop the guard next tick so any bindings/animations settle first
        Qt.callLater(function () {
            root._applyingConfigCameraTransform = false;
        });
    }

    function commitCameraTransformToConfig() {
        if (!root.cameraAdapter)
            return;

        if (root._applyingConfigCameraTransform)
            return;

        root._committing_config = true;
        Qt.callLater(function () {
            root._committing_config = false;
        });
        root.cameraAdapter.set_position(orbitOrigin.position);
        root.cameraAdapter.set_rotation(orbitOrigin.rotation);
        root.cameraAdapter.set_distance(camera.z);
    }

    function refreshFromAdapter() {
        cameraAdapter = sessionAdapter ? sessionAdapter.cameraAdapter : null;
        if (!cameraAdapter)
            return;
        // initial pull: config -> UI
        applyCameraTogglesFromConfig();
        applyCameraTransformFromConfig();
        // snap projection immediately on startup / adapter swap
        camera.setBlend(sessionControls.orthographicEnabled ? 1 : 0, false);
    }

    anchors.fill: parent
    Component.onCompleted: refreshFromAdapter()
    onSessionAdapterChanged: refreshFromAdapter()

    // config -> UI
    Connections {
        function onLockedChanged() {
            applyCameraTogglesFromConfig();
        }

        function onShow_gridChanged() {
            applyCameraTogglesFromConfig();
        }

        function onOrthographicChanged() {
            applyCameraTogglesFromConfig();
            // config-driven change gets no animation
            camera.setBlend(sessionControls.orthographicEnabled ? 1 : 0, false);
        }

        function onPositionChanged() {
            if (root._committing_config)
                return;

            applyCameraTransformFromConfig();
        }

        function onRotationChanged() {
            if (root._committing_config)
                return;

            applyCameraTransformFromConfig();
        }

        function onDistanceChanged() {
            if (root._committing_config)
                return;

            applyCameraTransformFromConfig();
        }

        target: root.cameraAdapter
        enabled: !!root.cameraAdapter
    }

    // UI -> config (toggles)
    Connections {
        function onViewLockedChanged() {
            if (!root.cameraAdapter || root._applyingConfigCameraToggles)
                return;

            const desired = !!sessionControls.viewLocked;
            const current = !!root.cameraAdapter.locked;
            if (desired === current)
                return;

            root.cameraAdapter.set_locked(desired);
        }

        function onGridEnabledChanged() {
            if (!root.cameraAdapter || root._applyingConfigCameraToggles)
                return;

            const desired = !!sessionControls.gridEnabled;
            const current = !!root.cameraAdapter.show_grid;
            if (desired === current)
                return;

            root.cameraAdapter.set_show_grid(desired);
        }

        function onOrthographicEnabledChanged() {
            // Animate only for user-driven toggles; config-driven updates should snap.
            const targetBlend = sessionControls.orthographicEnabled ? 1 : 0;
            camera.setBlend(targetBlend, !root._applyingConfigCameraToggles);
            if (!root.cameraAdapter || root._applyingConfigCameraToggles)
                return;

            const desired = !!sessionControls.orthographicEnabled;
            const current = !!root.cameraAdapter.orthographic;
            if (desired === current)
                return;

            root.cameraAdapter.set_orthographic(desired);
        }

        target: sessionControls
    }

    View3D {
        id: view

        property var selectedObject: null
        readonly property quaternion gizmoBasis: Qt.quaternion(1, 0, 0, 0)
        readonly property quaternion gizmoBasisInv: gizmoBasis.conjugated()

        function orbitToGizmoRotation(q) {
            return gizmoBasis.times(q).times(gizmoBasisInv);
        }

        function gizmoToOrbitRotation(q) {
            return gizmoBasisInv.times(q).times(gizmoBasis);
        }

        anchors.fill: parent
        camera: camera

        Node {
            id: selectionProxy
            x: selectionGizmo.dragging ? selectionGizmo.dragPosition.x : selectionPosition.x
            y: selectionGizmo.dragging ? selectionGizmo.dragPosition.y : selectionPosition.y
            z: selectionGizmo.dragging ? selectionGizmo.dragPosition.z : selectionPosition.z
            scale: selectionGizmo.dragging ? selectionGizmo.dragScale : selectionScale
            eulerRotation: selectionGizmo.dragging ? selectionGizmo.dragRotation : selectionRotation
        }

        Connections {
            target: selectedDeviceAdapter
            function onFieldChanged(path) {
                // TODO might want to debounce
                if (path.includes("transform")) {
                    root.selectionTransformUpdate();
                }
            }
        }

        Connections {
            target: selectedOperatorAdapter
            function onFieldChanged(path) {
                if (path.includes("camera")) {
                    root.selectionTransformUpdate();
                }
            }
        }

        Repeater3D {
            model: root.deviceAdapters

            Node {
                id: deviceNode

                property bool isSelected: index === root.workspace.selectedDeviceIndex && !root.selectedOperatorAdapter

                // TODO
                // visual offset during gizmo drag (translation only for now cause that's easier than figuring out how to update rotation origins lol)
                x: isSelected && selectionGizmo.dragging ? selectionGizmo.dragPosition.x - selectionPosition.x : 0
                y: isSelected && selectionGizmo.dragging ? selectionGizmo.dragPosition.y - selectionPosition.y : 0
                z: isSelected && selectionGizmo.dragging ? selectionGizmo.dragPosition.z - selectionPosition.z : 0

                Model {
                    id: model

                    property string model: modelData ? modelData.id + "_model" : ""
                    property string geo: model + "_geo"

                    pickable: true

                    geometry: PointCloudGeometry {
                        id: geo
                        pointCloudAdapter: modelData.pointCloudAdapter() ?? []
                        enabled: modelData.render
                    }

                    materials: [
                        PointCloudMaterial {
                            uPointSize: viewController.shaderPointSize
                        }
                    ]
                }

                Connections {
                    target: modelData
                    function onPointCloudUpdated() {
                        geo.updateGeometry();
                    }
                }
            }
        }

        environment: SceneEnvironment {
            clearColor: AppSettings.backgroundColor
            backgroundMode: SceneEnvironment.Color
            depthPrePassEnabled: false
            // fog: Fog { enabled: false }
            // antialiasingMode: SceneEnvironment.SSAA
        }

        // ---------- CAMERA ----------
        Node {
            id: orbitOrigin

            position: root.defaultOrbitOriginPosition
            rotation: root.defaultOrbitOriginRotation

            SessionCamera {
                id: camera
            }
        }

        OrbitViewController {
            id: viewController
            anchors.fill: parent
            camera: camera
            origin: orbitOrigin
            enabled: !sessionControls.orbitRotationRunning && !sessionControls.viewLocked
            onMouseHeldChanged: {
                if (mouseHeld || sessionControls.orbitRotationRunning)
                    return;

                // on release:
                root.commitCameraTransformToConfig();
            }
            onScrollingChanged: {
                if (scrolling || sessionControls.orbitRotationRunning)
                    return;
                root.commitCameraTransformToConfig();
            }
            onDragActiveChanged: {
                if (!dragActive) {
                    // Commit once when the interaction finishes.
                    root.commitCameraTransformToConfig();
                }
            }
        }

        // a node for camera orbit gizmo to bind to
        Node {
            id: gizmoTarget
        }

        Binding {
            target: gizmoTarget
            property: "rotation"
            value: view.orbitToGizmoRotation(camera.sceneRotation.conjugated())
        }

        // ---------- PICKING + FOCUS ----------
        TapHandler {
            target: view
            acceptedButtons: Qt.LeftButton
            onTapped: (p, b) => {
                const hit = view.pick(p.position.x, p.position.y).objectHit;
                view.selectedObject = hit || null;
            }
            onDoubleTapped: (p, b) => {
                if (sessionControls.viewLocked)
                    return;

                const result = view.pick(p.position.x, p.position.y);
                const hit = result.objectHit;
                if (!hit)
                    return;

                focusAnimation.from = orbitOrigin.position;
                focusAnimation.to = hit.position;
                focusAnimation.start();
            }
        }

        Model {
            id: groundGrid
            visible: sessionControls.gridEnabled

            scale: Qt.vector3d(1000, 1000, 0)
            eulerRotation: Qt.vector3d(90, 0, 0)

            property real metres: Math.round(AppSettings.gridSizeMetres)
            geometry: GridGeometry {
                horizontalLines: Math.round(groundGrid.metres + 1)
                verticalLines: Math.round(groundGrid.metres + 1)
            }

            materials: [
                PrincipledMaterial {
                    baseColor: ThemeColors.midlight
                    lighting: PrincipledMaterial.NoLighting
                    lineWidth: Math.round(Scaling.uiScale * 2)
                }
            ]
        }

        Model {
            id: xAxis
            visible: sessionControls.gridEnabled
            eulerRotation: Qt.vector3d(90, 0, 0)
            scale: Qt.vector3d(groundGrid.metres * 500, 1, 1)
            x: groundGrid.metres * 25

            geometry: GridGeometry {
                horizontalLines: 1
                verticalLines: 1
            }

            materials: [
                PrincipledMaterial {
                    baseColor: ThemeColors.red
                    lighting: PrincipledMaterial.NoLighting
                    lineWidth: Math.round(Scaling.uiScale * 4)
                }
            ]
        }

        Model {
            id: zAxis
            visible: sessionControls.gridEnabled
            eulerRotation: Qt.vector3d(90, 0, 00)
            scale: Qt.vector3d(1, groundGrid.metres * 500, 1)
            z: groundGrid.metres * 25

            geometry: GridGeometry {
                horizontalLines: 1
                verticalLines: 1
            }

            materials: [
                PrincipledMaterial {
                    baseColor: ThemeColors.blue
                    lighting: PrincipledMaterial.NoLighting
                    lineWidth: Math.round(Scaling.uiScale * 4)
                }
            ]
        }
    }

    // session window GUI overlaid on top of the View3D
    SessionControls {
        id: sessionControls
        workspace: root.workspace
        sessionView: root
        anchors.fill: parent
        view3d: view
        gizmoTarget: gizmoTarget
        orbitOrigin: orbitOrigin
        z: 120
        onRequestHomeCamera: {
            if (sessionControls.viewLocked)
                return;

            focusAnimation.stop();
            homeOrbitOriginAnim.stop();
            homeOrbitOriginPositionAnim.from = orbitOrigin.position;
            homeOrbitOriginRotationAnim.from = orbitOrigin.rotation;
            homeOrbitOriginDistanceAnim.from = camera.z;
            homeOrbitOriginAnim.start();
        }
    }

    // transform controls for selected device/operator

    SelectionGizmo {
        id: selectionGizmo
        visible: root.selectedOperatorAdapter || root.selectedDeviceAdapter && !sessionControls.viewLocked
        view3d: view
        targetNode: selectionProxy
        // TODO the mode should be based on what kind of transformation the adapter exposes
        mode: GizmoEnums.Mode.All
        targetAdapter: root.selectedOperatorAdapter || root.selectedDeviceAdapter
        cameraTarget: root.selectedOperatorAdapter !== null
        z: 99
    }

    // Camera move animations
    Vector3dAnimation {
        id: focusAnimation

        target: orbitOrigin
        property: "position"
        duration: 350
        easing.type: Easing.OutQuart
    }

    ParallelAnimation {
        id: homeOrbitOriginAnim
        alwaysRunToEnd: true
        onFinished: {
            orbitOrigin.position = root.defaultOrbitOriginPosition;
            orbitOrigin.rotation = root.defaultOrbitOriginRotation;
            camera.z = root.defaultCameraDistance;
            root.commitCameraTransformToConfig();
        }

        Vector3dAnimation {
            id: homeOrbitOriginPositionAnim
            to: root.defaultOrbitOriginPosition
            target: orbitOrigin
            property: "position"
            duration: 420
            easing.type: Easing.OutCubic
        }

        PropertyAnimation {
            id: homeOrbitOriginRotationAnim
            to: root.defaultOrbitOriginRotation
            target: orbitOrigin
            property: "rotation"
            duration: 420
            easing.type: Easing.OutCubic
        }

        NumberAnimation {
            id: homeOrbitOriginDistanceAnim
            to: root.defaultCameraDistance
            target: camera
            property: "z"
            duration: 420
            easing.type: Easing.OutCubic
        }
    }

    // view border is inset on top of content
    Rectangle {
        anchors.fill: view
        color: "transparent"
        border.width: root.showBorder ? 1 : 0
        border.color: root.borderColor
        z: 99
        enabled: false
    }
}
