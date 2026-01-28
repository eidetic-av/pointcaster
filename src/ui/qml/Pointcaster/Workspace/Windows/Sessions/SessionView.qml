import QtQuick
import QtQuick.Controls
import QtQuick3D
import QtQuick3D.Helpers

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root
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

    View3D {
        id: view
        anchors.fill: parent
        camera: camera

        property var selectedObject: null

        property bool gridEnabled: true

        environment: SceneEnvironment {
            clearColor: ThemeColors.shadow
            backgroundMode: SceneEnvironment.Color

            InfiniteGrid {
                id: groundGrid
                gridInterval: 100
                gridAxes: true
                visible: view.gridEnabled
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

            PerspectiveCamera {
                id: camera
                z: root.defaultCameraDistance
            }
        }

        // A real Node for the gizmo to bind to
        Node {
            id: gizmoTarget
        }

        // Flip 180° about X
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

        // RMB drag = pan (1/2)
        DragHandler {
            acceptedButtons: Qt.RightButton
            enabled: !sessionCameraControls.viewLocked

            property real lastX: 0
            property real lastY: 0

            onActiveChanged: {
                if (active) {
                    lastX = translation.x;
                    lastY = translation.y;
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

        // RMB drag = pan (2/2) (kept as requested)
        DragHandler {
            acceptedButtons: Qt.RightButton
            enabled: !sessionCameraControls.viewLocked

            onTranslationChanged: {
                const panScale = Math.abs(camera.z) * 0.002;

                const sceneDelta = Qt.vector3d(camera.right.x, camera.right.y, camera.right.z).times(-translation.x * panScale).plus(Qt.vector3d(camera.up.x, camera.up.y, camera.up.z).times(translation.y * panScale));

                const localDelta = orbitOrigin.mapDirectionFromScene(sceneDelta);

                orbitOrigin.position = Qt.vector3d(orbitOrigin.position.x + localDelta.x, orbitOrigin.position.y + localDelta.y, orbitOrigin.position.z + localDelta.z);

                translation = Qt.point(0, 0);
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

        Vector3dAnimation {
            id: focusAnimation
            target: orbitOrigin
            property: "position"
            duration: 350
            easing.type: Easing.OutQuart
        }
    }

    // ---------------- CAMERA CONTROLS OVERLAY ----------------
    SessionCameraControls {
        id: sessionCameraControls
        view3d: view
        gizmoTarget: gizmoTarget
        orbitOrigin: orbitOrigin
        z: 120

        // drives grid visibility
        onGridEnabledChanged: view.gridEnabled = gridEnabled

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

    Vector3dAnimation {
        id: homeOrbitOriginPositionAnim
        target: orbitOrigin
        property: "position"
        duration: 420
        easing.type: Easing.OutCubic
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
        radius: 0
        z: 99
        enabled: false
    }
}
