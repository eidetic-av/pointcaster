import QtQuick
import QtQuick3D
import QtQuick3D.Helpers
import Pointcaster 1.0

Item {
    id: root

    required property QtObject deviceAdapter
    required property bool live
    required property bool enablePicking
    required property bool active
    property var markerPositions: []

    signal picked(vector3d position)

    // index-encoded view for picking
    View3D {
        id: pickView
        anchors.fill: parent
        renderMode: View3D.Offscreen
        visible: root.enablePicking

        Model {
            geometry: geo
            materials: [
                PointIndexMaterial {
                    uPointSize: viewController.shaderPointSize
                }
            ]
        }

        PerspectiveCamera {
            id: pickCamera
            position: cameraNode.scenePosition
            rotation: cameraNode.sceneRotation
            fieldOfView: cameraNode.fieldOfView
            clipNear: cameraNode.clipNear
            clipFar: cameraNode.clipFar
        }

        environment: SceneEnvironment {
            clearColor: Qt.rgba(1, 1, 1, 1)
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.NoAA
            tonemapMode: SceneEnvironment.TonemapModeNone
        }
    }

    // regular color point cloud view
    View3D {
        id: cloudView
        anchors.fill: parent
        camera: cameraNode

        Model {
            geometry: PointCloudGeometry {
                id: geo
                pointCloudAdapter: root.deviceAdapter ? root.deviceAdapter.pointCloudAdapter() : null
            }
            materials: [
                PointCloudMaterial {
                    uPointSize: viewController.shaderPointSize
                }
            ]
        }

        Node {
            id: orbitOrigin
            position: geo.boundsCenter

            PerspectiveCamera {
                id: cameraNode
                z: 150
            }
        }

        Node {
            id: markerContainer
        }

        OrbitViewController {
            id: viewController
            anchors.fill: parent
            camera: cameraNode
            origin: orbitOrigin
        }

        environment: SceneEnvironment {
            clearColor: ThemeColors.shadow
            backgroundMode: SceneEnvironment.Color
        }
    }

    // rebuild markers whenever positions change
    onMarkerPositionsChanged: rebuildMarkers()

    function rebuildMarkers() {
        // destroy existing
        for (var i = markerContainer.children.length - 1; i >= 0; --i)
            markerContainer.children[i].destroy();

        for (var j = 0; j < markerPositions.length; ++j) {
            markerComponent.createObject(markerContainer, {
                position: markerPositions[j]
            });
        }
    }

    Component {
        id: markerComponent

        Model {
            source: "#Sphere"
            scale: Qt.vector3d(0.02, 0.02, 0.02)
            materials: [
                PrincipledMaterial {
                    baseColor: "red"
                    lighting: PrincipledMaterial.NoLighting
                }
            ]
        }
    }

    Connections {
        target: root.deviceAdapter
        function onPointCloudUpdated() {
            if (root.live)
                geo.updateGeometry();
        }
    }

    MouseArea {
        anchors.fill: parent
        propagateComposedEvents: true

        onDoubleClicked: mouse => {
            if (!root.active || !root.enablePicking)
                return;
            colorPicker.pick(pickView, Math.round(mouse.x), Math.round(mouse.y));
        }
    }

    ColorPicker {
        id: colorPicker

        onPicked: {
            const index = ((colorPicker.r << 24) | (colorPicker.g << 16) | (colorPicker.b << 8) | colorPicker.a) >>> 0;
            if (index === 0xFFFFFFFF)
                return;

            const pos = geo.pointPosition(index);
            console.log("picked point index:", index, "position:", pos);
            root.picked(pos);
        }
    }

    // active highlight border
    Rectangle {
        anchors.fill: parent
        color: "transparent"
        border.width: root.active ? Math.round(Scaling.uiScale * 2) : 0
        border.color: ThemeColors.highlight
        visible: root.active
    }

    function snapshot() {
        geo.updateGeometry();
        orbitOrigin.position = geo.boundsCenter;
    }

    function reset() {
        geo.reset();
        orbitOrigin.position = {};
        markerPositions = [];
    }
}
