import QtQuick
import QtQuick.Controls
import QtQuick3D
import QtQuick3D.Helpers

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    required property var sessionAdapter
    property var cameraAdapter: null

    anchors.fill: parent

    property bool showBorder: false
    property color borderColor: ThemeColors.highlight

    readonly property real defaultCameraDistance: 500
    readonly property vector3d defaultOrbitOriginPosition: Qt.vector3d(0, 0, 0)
    readonly property quaternion defaultOrbitOriginRotation: {
        const pitch = -17;
        const yaw = 0;

        const pitchRad = pitch * Math.PI / 180.0;
        const yawRad = yaw * Math.PI / 180.0;

        const qPitch = Qt.quaternion(Math.cos(pitchRad * 0.5), Math.sin(pitchRad * 0.5), 0, 0);
        const qYaw = Qt.quaternion(Math.cos(yawRad * 0.5), 0, Math.sin(yawRad * 0.5), 0);
        return qYaw.times(qPitch);
    }

    // Guards to prevent config->UI updates from immediately re-committing UI->config (undo/redo feedback loops)
    property bool _applyingConfigCameraPosition: false
    property bool _applyingConfigCameraToggles: false

    function _vector3dFromAdapterPosition(p) {
        // p is expected to be QVector3D-ish in QML (has x/y/z)
        if (!p)
            return Qt.vector3d(0, 0, 0);
        return Qt.vector3d(Number(p.x) || 0, Number(p.y) || 0, Number(p.z) || 0);
    }

    function applyCameraPositionFromConfig() {
        if (!root.cameraAdapter)
            return;

        root._applyingConfigCameraPosition = true;
        orbitOrigin.position = _vector3dFromAdapterPosition(root.cameraAdapter.position);

        // drop the guard next tick so any bindings/animations settle first
        Qt.callLater(function () {
            root._applyingConfigCameraPosition = false;
        });
    }

    function applyCameraTogglesFromConfig() {
        if (!root.cameraAdapter)
            return;

        root._applyingConfigCameraToggles = true;

        sessionCameraControls.viewLocked = !!root.cameraAdapter.locked;
        sessionCameraControls.gridEnabled = !!root.cameraAdapter.show_grid;
        sessionCameraControls.orthographicEnabled = !!root.cameraAdapter.orthographic;

        // drop the guard next tick so any bindings/animations settle first
        Qt.callLater(function () {
            root._applyingConfigCameraToggles = false;
        });
    }

    function commitCameraPositionToConfig() {
        if (!root.cameraAdapter)
            return;
        if (root._applyingConfigCameraPosition)
            return;

        // QML vector3d -> C++ QVariant should marshal to QVector3D
        root.cameraAdapter.set_position(orbitOrigin.position);
    }

    function refreshFromAdapter() {
        cameraAdapter = sessionAdapter ? sessionAdapter.cameraAdapter : null;
        if (!cameraAdapter)
            return;

        // initial pull: config -> UI
        applyCameraTogglesFromConfig();
        applyCameraPositionFromConfig();

        // snap projection immediately on startup / adapter swap
        camera.setBlend(sessionCameraControls.orthographicEnabled ? 1.0 : 0.0, false);
    }

    Component.onCompleted: refreshFromAdapter()
    onSessionAdapterChanged: refreshFromAdapter()

    // config -> UI
    Connections {
        target: root.cameraAdapter
        enabled: !!root.cameraAdapter

        function onLockedChanged() {
            applyCameraTogglesFromConfig();
        }

        function onShow_gridChanged() {
            applyCameraTogglesFromConfig();
        }

        function onOrthographicChanged() {
            applyCameraTogglesFromConfig();

            // config-driven change gets no animation
            camera.setBlend(sessionCameraControls.orthographicEnabled ? 1.0 : 0.0, false);
        }

        function onPositionChanged() {
            applyCameraPositionFromConfig();
        }
    }

    // UI -> config (toggles)
    Connections {
        target: sessionCameraControls

        function onViewLockedChanged() {
            if (!root.cameraAdapter || root._applyingConfigCameraToggles)
                return;

            const desired = !!sessionCameraControls.viewLocked;
            const current = !!root.cameraAdapter.locked;
            if (desired === current)
                return;

            root.cameraAdapter.set_locked(desired);
        }

        function onGridEnabledChanged() {
            if (!root.cameraAdapter || root._applyingConfigCameraToggles)
                return;

            const desired = !!sessionCameraControls.gridEnabled;
            const current = !!root.cameraAdapter.show_grid;
            if (desired === current)
                return;

            root.cameraAdapter.set_show_grid(desired);
        }

        function onOrthographicEnabledChanged() {
            // Animate only for user-driven toggles; config-driven updates should snap.
            const targetBlend = sessionCameraControls.orthographicEnabled ? 1.0 : 0.0;
            camera.setBlend(targetBlend, !root._applyingConfigCameraToggles);

            if (!root.cameraAdapter || root._applyingConfigCameraToggles)
                return;

            const desired = !!sessionCameraControls.orthographicEnabled;
            const current = !!root.cameraAdapter.orthographic;
            if (desired === current)
                return;

            root.cameraAdapter.set_orthographic(desired);
        }
    }

    View3D {
        id: view
        anchors.fill: parent
        camera: camera

        property var selectedObject: null

        environment: SceneEnvironment {
            clearColor: ThemeColors.shadow
            backgroundMode: SceneEnvironment.Color

            InfiniteGrid {
                id: groundGrid
                gridInterval: 100
                gridAxes: true
                visible: sessionCameraControls.gridEnabled
            }
        }

        // -------- LIGHT RIG --------
        DirectionalLight {
            eulerRotation: Qt.vector3d(-35, 35, 0)
            brightness: 55
            ambientColor: Qt.rgba(0.18, 0.18, 0.18, 1.0)
        }
        DirectionalLight {
            eulerRotation: Qt.vector3d(-5, -120, 0)
            brightness: 40
            ambientColor: Qt.rgba(0.26, 0.26, 0.26, 1.0)
        }
        DirectionalLight {
            eulerRotation: Qt.vector3d(25, 160, 0)
            brightness: 45
            ambientColor: Qt.rgba(0.20, 0.20, 0.20, 1.0)
        }

        // ---------- CAMERA ----------
        Node {
            id: orbitOrigin
            position: root.defaultOrbitOriginPosition
            rotation: root.defaultOrbitOriginRotation

            CustomCamera {
                id: camera
                // 0 = fully perspective, 1 = fully orthographic
                property real blend: 0.0

                z: root.defaultCameraDistance

                // Perspective params
                property real nearPlane: 50
                property real farPlane: 15000
                property real fovYRadians: 60.0 * Math.PI / 180.0

                // ortho params
                property real orthoHalfHeight: z * 0.6

                // OrbitCameraController expects these to exist
                property real clipNear
                property real clipFar

                projection: {
                    const aspect = view.width > 0 ? (view.width / view.height) : 1.0;
                    const t = blend;

                    // --- Perspective ---
                    const cot = Math.cos(fovYRadians * 0.5) / Math.sin(fovYRadians * 0.5);
                    const p00 = cot / aspect;
                    const p11 = cot;
                    const p22 = -(nearPlane + farPlane) / (farPlane - nearPlane);
                    const p23 = -(2.0 * nearPlane * farPlane) / (farPlane - nearPlane);
                    const p32 = -1.0;
                    const p33 = 0.0;

                    // --- Orthographic ---
                    const top = orthoHalfHeight;
                    const bottom = -orthoHalfHeight;
                    const right = orthoHalfHeight * aspect;
                    const left = -orthoHalfHeight * aspect;

                    const o00 = 2.0 / (right - left);
                    const o11 = 2.0 / (top - bottom);
                    const o22 = -2.0 / (farPlane - nearPlane);
                    const o23 = -(farPlane + nearPlane) / (farPlane - nearPlane);
                    const o32 = 0.0;
                    const o33 = 1.0;

                    const m00 = lerp(p00, o00, t);
                    const m11 = lerp(p11, o11, t);
                    const m22 = lerp(p22, o22, t);
                    const m23 = lerp(p23, o23, t);
                    const m32 = lerp(p32, o32, t);
                    const m33 = lerp(p33, o33, t);

                    return Qt.matrix4x4(m00, 0, 0, 0, 0, m11, 0, 0, 0, 0, m22, m23, 0, 0, m32, m33);
                }

                function lerp(a, b, t) {
                    return a + (b - a) * t;
                }

                function setBlend(targetValue, animate) {
                    projectionBlendAnim.stop();

                    if (!animate) {
                        blend = targetValue;
                        return;
                    }

                    projectionBlendAnim.from = blend;
                    projectionBlendAnim.to = targetValue;

                    // easing/duration differs by direction because the matrix blend is non-linear
                    const ascending = projectionBlendAnim.to > projectionBlendAnim.from;
                    projectionBlendAnim.easing.type = ascending ? Easing.OutExpo : Easing.InCubic;
                    projectionBlendAnim.duration = ascending ? 150 : 350;

                    projectionBlendAnim.start();
                }

                NumberAnimation {
                    id: projectionBlendAnim
                    target: camera
                    property: "blend"
                    // duration/easing set in setBlend()
                }
            }
        }

        // A real Node for the gizmo to bind to
        Node {
            id: gizmoTarget
        }

        readonly property quaternion gizmoBasis: Qt.quaternion(1, 0, 0, 0)
        readonly property quaternion gizmoBasisInv: gizmoBasis.conjugated()

        function orbitToGizmoRotation(q) {
            return gizmoBasis.times(q).times(gizmoBasisInv);
        }
        function gizmoToOrbitRotation(q) {
            return gizmoBasisInv.times(q).times(gizmoBasis);
        }

        Binding {
            target: gizmoTarget
            property: "rotation"
            value: view.orbitToGizmoRotation(camera.sceneRotation.conjugated())
        }

        OrbitCameraController {
            anchors.fill: parent
            camera: camera
            origin: orbitOrigin
            acceptedButtons: Qt.LeftButton
            xSpeed: 0.5
            ySpeed: 0.5
            enabled: !sessionCameraControls.orbitRotationRunning && !sessionCameraControls.viewLocked
        }

        // RMB drag = pan
        DragHandler {
            acceptedButtons: Qt.RightButton
            enabled: !sessionCameraControls.viewLocked

            property real lastX: 0
            property real lastY: 0

            onActiveChanged: {
                if (active) {
                    lastX = translation.x;
                    lastY = translation.y;
                } else {
                    // Commit once when the interaction finishes.
                    root.commitCameraPositionToConfig();
                }
            }

            onTranslationChanged: {
                const dx = translation.x - lastX;
                const dy = translation.y - lastY;
                lastX = translation.x;
                lastY = translation.y;

                const panScale = Math.abs(camera.z) * 0.002;
                const sceneDelta = Qt.vector3d(camera.right.x, camera.right.y, camera.right.z).times(-dx * panScale).plus(Qt.vector3d(camera.up.x, camera.up.y, camera.up.z).times(dy * panScale));

                const parentNode = orbitOrigin.parent;
                const parentDelta = parentNode && parentNode.mapDirectionFromScene ? parentNode.mapDirectionFromScene(sceneDelta) : sceneDelta;

                orbitOrigin.position = Qt.vector3d(orbitOrigin.position.x + parentDelta.x, orbitOrigin.position.y + parentDelta.y, orbitOrigin.position.z + parentDelta.z);
            }
        }

        // ---------- MODELS ----------
        Model {
            id: cubeMain
            source: "#Cube"
            pickable: true
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.FragmentLighting
                baseColor: view.selectedObject === cubeMain ? ThemeColors.highlight : "red"
                roughness: 0.85
                metalness: 0.0
                opacity: 0.5
            }
        }

        Model {
            id: cubeSmall
            source: "#Cube"
            pickable: true
            x: -150
            y: 60
            scale: Qt.vector3d(0.5, 0.5, 0.5)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.FragmentLighting
                baseColor: view.selectedObject === cubeSmall ? ThemeColors.highlight : "deepskyblue"
                roughness: 0.85
                metalness: 0.0
                opacity: 0.5
            }
        }

        Model {
            id: cubeWide
            source: "#Cube"
            pickable: true
            x: 180
            y: 20
            scale: Qt.vector3d(1.5, 0.5, 0.5)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.FragmentLighting
                baseColor: view.selectedObject === cubeWide ? ThemeColors.highlight : "limegreen"
                roughness: 0.85
                metalness: 0.0
                opacity: 0.5
            }
        }

        Model {
            id: cubeTall
            source: "#Cube"
            pickable: true
            z: -150
            y: -40
            scale: Qt.vector3d(0.6, 1.8, 0.6)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.FragmentLighting
                baseColor: view.selectedObject === cubeTall ? ThemeColors.highlight : "orange"
                roughness: 0.85
                metalness: 0.0
                opacity: 0.5
            }
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
                if (sessionCameraControls.viewLocked)
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
    }

    // ---------------- CAMERA CONTROLS OVERLAY ----------------
    SessionCameraControls {
        id: sessionCameraControls
        view3d: view
        gizmoTarget: gizmoTarget
        orbitOrigin: orbitOrigin
        z: 120

        onRequestHomeCamera: {
            if (sessionCameraControls.viewLocked)
                return;

            focusAnimation.stop();
            homeOrbitOriginPositionAnim.stop();
            homeOrbitOriginRotationAnim.stop();
            homeCameraZAnim.stop();

            homeOrbitOriginPositionAnim.from = orbitOrigin.position;
            homeOrbitOriginPositionAnim.to = root.defaultOrbitOriginPosition;
            homeOrbitOriginPositionAnim.start();

            homeOrbitOriginRotationAnim.from = orbitOrigin.rotation;
            homeOrbitOriginRotationAnim.to = root.defaultOrbitOriginRotation;
            homeOrbitOriginRotationAnim.start();

            camera.x = 0;
            camera.y = 0;
            homeCameraZAnim.from = camera.z;
            homeCameraZAnim.to = root.defaultCameraDistance;
            homeCameraZAnim.start();
        }
    }

    // Camera move animations
    Vector3dAnimation {
        id: focusAnimation
        target: orbitOrigin
        property: "position"
        duration: 350
        easing.type: Easing.OutQuart
        onStopped: root.commitCameraPositionToConfig()
    }
    Vector3dAnimation {
        id: homeOrbitOriginPositionAnim
        target: orbitOrigin
        property: "position"
        duration: 420
        easing.type: Easing.OutCubic
        onStopped: root.commitCameraPositionToConfig()
    }
    PropertyAnimation {
        id: homeOrbitOriginRotationAnim
        target: orbitOrigin
        property: "rotation"
        duration: 420
        easing.type: Easing.OutCubic
    }
    NumberAnimation {
        id: homeCameraZAnim
        target: camera
        property: "z"
        duration: 420
        easing.type: Easing.OutCubic
    }

    // border overlay
    Rectangle {
        anchors.fill: view
        color: "transparent"
        border.width: root.showBorder ? 1 : 0
        border.color: root.borderColor
        z: 99
        enabled: false
    }
}
