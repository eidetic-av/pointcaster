import QtQuick
import QtQuick.Controls
import QtQuick3D
import QtQuick3D.Helpers

import Pointcaster 1.0
import Pointcaster.Workspace 1.0
import Pointcaster.Geometry 1.0

Item {
    id: root

    required property var workspace
    required property var sessionAdapter
    required property var deviceAdapters
    property var cameraAdapter: null

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
        // Model {
        //     id: cubeWide
        //     source: "#Cube"
        //     pickable: true
        //     x: 180
        //     y: 20
        //     scale: Qt.vector3d(1.5, 0.5, 0.5)
        //     materials: PrincipledMaterial {
        //         lighting: PrincipledMaterial.FragmentLighting
        //         baseColor: view.selectedObject === cubeWide ? ThemeColors.highlight : "limegreen"
        //         roughness: 0.85
        //         metalness: 0.0
        //         opacity: 0.5
        //     }
        // }
        // Model {
        //     id: cubeTall
        //     source: "#Cube"
        //     pickable: true
        //     z: -150
        //     y: -40
        //     scale: Qt.vector3d(0.6, 1.8, 0.6)
        //     materials: PrincipledMaterial {
        //         lighting: PrincipledMaterial.FragmentLighting
        //         baseColor: view.selectedObject === cubeTall ? ThemeColors.highlight : "orange"
        //         roughness: 0.85
        //         metalness: 0.0
        //         opacity: 0.5
        //     }
        // }
        // Device point clouds

        id: view
        // ---------- MODELS ----------
        //     id: cubeMain
        //     source: "#Cube"
        //     pickable: true
        //     materials: PrincipledMaterial {
        //         lighting: PrincipledMaterial.FragmentLighting
        //         baseColor: view.selectedObject === cubeMain ? ThemeColors.highlight : "red"
        //         roughness: 0.85
        //         metalness: 0.0
        //         opacity: 0.5
        //     }
        // }
        Model {
            id: cubeTest
            source: "#Cube"
            pickable: true

            property var selectedDeviceAdapter: (root.workspace && root.deviceAdapters) ? root.deviceAdapters[root.workspace.selectedDeviceIndex] : null

            property var devicePos: selectedDeviceAdapter ? selectedDeviceAdapter.value("transform/position") : Qt.vector3d(0, 0, 0)
            property var deviceScale: selectedDeviceAdapter ? selectedDeviceAdapter.value("transform/scale") : Qt.vector3d(1, 1, 1)

            x: devicePos ? devicePos.x * 100 : 0
            y: devicePos ? devicePos.y * 100 : 0
            z: devicePos ? devicePos.z * 100 : 0

            scale: deviceScale || Qt.vector3d(1, 1, 1)

            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: "deepskyblue"
                roughness: 0.85
                metalness: 0.0
                opacity: 0.5
            }
        }

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

        Repeater3D {
            model: root.deviceAdapters

            Node {
                Model {
                    id: model

                    property string model: modelData ? "device_" + modelData.deviceIndex : ""
                    property string geo: "geo_" + model

                    pickable: true

                    geometry: PointCloudGeometry {
                        id: geo
                        pointCloudAdapter: modelData.pointCloudAdapter() ?? []
                    }

                    materials: [
                        PrincipledMaterial {
                            lighting: PrincipledMaterial.NoLighting
                            pointSize: Math.round(2 * Scaling.uiScale)
                            // TODO this pointSize needs to change based on the distance of the point to the camera...
                            // we can achieve this with a custom vertex and/or fragment shader
                            // CustomMaterial {
                            // vertexShader: "material.vert"
                            // fragmentShader: "material.frag"
                        }
                    ]
                }

                Connections {
                    target: modelData
                    function onPointCloudUpdated() {
                        geo.updateGeometry()
                    }
                }
            }
        }

        // -------- TESTS -------

        // // -------- LIGHT RIG --------
        // DirectionalLight {
        //     eulerRotation: Qt.vector3d(-35, 35, 0)
        //     brightness: 55
        //     ambientColor: Qt.rgba(0.18, 0.18, 0.18, 1)
        // }

        // DirectionalLight {
        //     eulerRotation: Qt.vector3d(-5, -120, 0)
        //     brightness: 40
        //     ambientColor: Qt.rgba(0.26, 0.26, 0.26, 1)
        // }

        // DirectionalLight {
        //     eulerRotation: Qt.vector3d(25, 160, 0)
        //     brightness: 45
        //     ambientColor: Qt.rgba(0.2, 0.2, 0.2, 1)
        // }

        environment: SceneEnvironment {
            clearColor: ThemeColors.shadow
            backgroundMode: SceneEnvironment.Color
            depthPrePassEnabled: true
            // fog: Fog { enabled: false }
            // antialiasingMode: SceneEnvironment.SSAA
        }

        // ---------- CAMERA ----------
        Node {
            id: orbitOrigin

            position: root.defaultOrbitOriginPosition
            rotation: root.defaultOrbitOriginRotation

            CustomCamera {
                id: camera

                // 0 = fully perspective, 1 = fully orthographic
                property real blend: 0
                // Perspective params
                property real nearPlane: 1
                property real farPlane: 250000
                property real fovYRadians: 60 * Math.PI / 180
                // ortho params
                property real orthoHalfHeight: z * 0.6
                // OrbitCameraController expects these to exist
                property real clipNear
                property real clipFar

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

                z: root.defaultCameraDistance
                projection: {
                    const aspect = view.width > 0 ? (view.width / view.height) : 1;
                    const t = blend;
                    // --- Perspective ---
                    const cot = Math.cos(fovYRadians * 0.5) / Math.sin(fovYRadians * 0.5);
                    const p00 = cot / aspect;
                    const p11 = cot;
                    const p22 = -(nearPlane + farPlane) / (farPlane - nearPlane);
                    const p23 = -(2 * nearPlane * farPlane) / (farPlane - nearPlane);
                    const p32 = -1;
                    const p33 = 0;
                    // --- Orthographic ---
                    const top = orthoHalfHeight;
                    const bottom = -orthoHalfHeight;
                    const right = orthoHalfHeight * aspect;
                    const left = -orthoHalfHeight * aspect;
                    const o00 = 2 / (right - left);
                    const o11 = 2 / (top - bottom);
                    // handles clipping dist, make it pretty much infinite for ortho
                    const o22 = -1e-08;
                    const o23 = 0;
                    const o32 = 0;
                    const o33 = 1;
                    const m00 = lerp(p00, o00, t);
                    const m11 = lerp(p11, o11, t);
                    const m22 = lerp(p22, o22, t);
                    const m23 = lerp(p23, o23, t);
                    const m32 = lerp(p32, o32, t);
                    const m33 = lerp(p33, o33, t);
                    return Qt.matrix4x4(m00, 0, 0, 0, 0, m11, 0, 0, 0, 0, m22, m23, 0, 0, m32, m33);
                }

                NumberAnimation {
                    // duration/easing set in setBlend()

                    id: projectionBlendAnim

                    target: camera
                    property: "blend"
                }
            }
        }

        // A real Node for the gizmo to bind to
        Node {
            id: gizmoTarget
        }

        Binding {
            target: gizmoTarget
            property: "rotation"
            value: view.orbitToGizmoRotation(camera.sceneRotation.conjugated())
        }

        OrbitViewController {
            anchors.fill: parent
            camera: camera
            origin: orbitOrigin
            acceptedButtons: Qt.LeftButton
            xSpeed: 0.1
            ySpeed: 0.5
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
        }

        // RMB drag = pan
        DragHandler {
            property real lastX: 0
            property real lastY: 0

            acceptedButtons: Qt.RightButton
            enabled: !sessionControls.viewLocked
            onActiveChanged: {
                if (active) {
                    lastX = translation.x;
                    lastY = translation.y;
                } else {
                    // Commit once when the interaction finishes.
                    root.commitCameraTransformToConfig();
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
            scale: Qt.vector3d(groundGrid.metres * 500, 0, 0)
            x: groundGrid.metres * 25

            geometry: GridGeometry {
                horizontalLines: 2
                verticalLines: 2
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
            scale: Qt.vector3d(0, groundGrid.metres * 500, 0)
            z: groundGrid.metres * 25

            geometry: GridGeometry {
                horizontalLines: 2
                verticalLines: 2
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
