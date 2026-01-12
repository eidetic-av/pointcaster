import QtQuick
import QtQuick.Controls
import QtQuick3D
import QtQuick3D.Helpers

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

View3D {
     id: view
     anchors.fill: parent
     camera: camera

     property var selectedObject: null
         readonly property color highlightColor: "#3af2ff"

             environment: SceneEnvironment {
                 clearColor: "black"
                 backgroundMode: SceneEnvironment.Color

                 InfiniteGrid {
                     id: groundGrid
                     gridInterval: 100
                     gridAxes: true
                     visible: true
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
    PerspectiveCamera { id: camera; z: 500 }
}

OrbitCameraController {
    anchors.fill: parent
    camera: camera
    origin: orbitOrigin
    acceptedButtons: Qt.LeftButton
    xSpeed: 0.5
    ySpeed: 0.5
}

DragHandler {
    acceptedButtons: Qt.RightButton

    property real lastX: 0
    property real lastY: 0

    onActiveChanged: {
        if (active) {
            lastX = translation.x
            lastY = translation.y
        }
    }

    onTranslationChanged: {
        const dx = translation.x - lastX
        const dy = translation.y - lastY
        lastX = translation.x
        lastY = translation.y

        const panScale = Math.abs(camera.z) * 0.002

        // scene-space delta in camera view plane (follows camera look direction)
        const sceneDelta =
              Qt.vector3d(camera.right.x, camera.right.y, camera.right.z).times(-dx * panScale)
            .plus(Qt.vector3d(camera.up.x,    camera.up.y,    camera.up.z   ).times( dy * panScale))

        // convert scene delta into orbitOrigin *parent* space (position is in parent space)
        const parentNode = orbitOrigin.parent
        const parentDelta = parentNode && parentNode.mapDirectionFromScene
            ? parentNode.mapDirectionFromScene(sceneDelta)
            : sceneDelta

        orbitOrigin.position = Qt.vector3d(
            orbitOrigin.position.x + parentDelta.x,
            orbitOrigin.position.y + parentDelta.y,
            orbitOrigin.position.z + parentDelta.z
        )
    }
}



// RMB drag = pan
DragHandler {
    acceptedButtons: Qt.RightButton

    onTranslationChanged: {
        // scale with orbit radius so it feels consistent
        const panScale = Math.abs(camera.z) * 0.002

        const sceneDelta =
              Qt.vector3d(camera.right.x, camera.right.y, camera.right.z).times(-translation.x * panScale)
            .plus(Qt.vector3d(camera.up.x,    camera.up.y,    camera.up.z   ).times( translation.y * panScale))

        const localDelta = orbitOrigin.mapDirectionFromScene(sceneDelta)

        orbitOrigin.position = Qt.vector3d(
            orbitOrigin.position.x + localDelta.x,
            orbitOrigin.position.y + localDelta.y,
            orbitOrigin.position.z + localDelta.z
        )

        // consume the delta so we apply incrementally
        translation = Qt.point(0, 0)
    }
}


             // ---------- MODELS ----------

             Model {
                 id: cubeMain
                 source: "#Cube"
                 pickable: true
                 materials: PrincipledMaterial {
                     lighting: PrincipledMaterial.FragmentLighting
                     baseColor: view.selectedObject === cubeMain ? view.highlightColor : "red"
                     roughness: 0.85
                     metalness: 0.0
                     opacity: 0.5
                 }
             }

             Model {
                 id: cubeSmall
                 source: "#Cube"
                 pickable: true
                 x: -150; y: 60
                 scale: Qt.vector3d(0.5, 0.5, 0.5)
                 materials: PrincipledMaterial {
                     lighting: PrincipledMaterial.FragmentLighting
                     baseColor: view.selectedObject === cubeSmall ? view.highlightColor : "deepskyblue"
                     roughness: 0.85
                     metalness: 0.0
                     opacity: 0.5
                 }
             }

             Model {
                 id: cubeWide
                 source: "#Cube"
                 pickable: true
                 x: 180; y: 20
                 scale: Qt.vector3d(1.5, 0.5, 0.5)
                 materials: PrincipledMaterial {
                     lighting: PrincipledMaterial.FragmentLighting
                     baseColor: view.selectedObject === cubeWide ? view.highlightColor : "limegreen"
                     roughness: 0.85
                     metalness: 0.0
                     opacity: 0.5
                 }
             }

             Model {
                 id: cubeTall
                 source: "#Cube"
                 pickable: true
                 z: -150; y: -40
                 scale: Qt.vector3d(0.6, 1.8, 0.6)
                 materials: PrincipledMaterial {
                     lighting: PrincipledMaterial.FragmentLighting
                     baseColor: view.selectedObject === cubeTall ? view.highlightColor : "orange"
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
                 const hit = view.pick(p.position.x, p.position.y).objectHit
                 view.selectedObject = hit || null
             }

             onDoubleTapped: (p, b) => {
             const result = view.pick(p.position.x, p.position.y)
             const hit = result.objectHit
             if (!hit) return
             focusAnimation.from = orbitOrigin.position
             focusAnimation.to = hit.position
             focusAnimation.start()
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