import QtQuick
import QtQuick3D
import QtQuick3D.Helpers

import Pointcaster 1.0

Item {
    id: root

    required property QtObject deviceAdapter

    property list<Model> markers: []

    View3D {
        id: mainView
        anchors.fill: parent
        camera: cameraNode

        Model {
            id: pointcloud

            geometry: PointCloudGeometry {
                id: geo
                pointCloudAdapter: root.deviceAdapter ? root.deviceAdapter.pointCloudAdapter() : null
            }

            materials: [
                PointIndexMaterial {
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
            clearColor: Qt.rgba(1, 1, 1, 1)
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.NoAA
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

    MouseArea {
        anchors.fill: parent
        propagateComposedEvents: true
        onClicked: mouse => colorPicker.pick(mainView, Math.round(mouse.x), Math.round(mouse.y))
    }

    ColorPicker {
        id: colorPicker
        onPicked: {
            const index = ((colorPicker.r << 24) | (colorPicker.g << 16) | (colorPicker.b << 8) | colorPicker.a) >>> 0;
            if (index === 0xFFFFFFFF) {
                // same as environment.clearColor (background)
                return;
            }
            const pos = geo.pointPosition(index);
            console.log("picked point index:", index, "position:", pos);
            var marker = markerComponent.createObject(markerContainer, {
                position: pos
            });
            markers.push(marker);
        }
    }

    function snapshot() {
        geo.updateGeometry();
        orbitOrigin.position = geo.boundsCenter;
    }
}
