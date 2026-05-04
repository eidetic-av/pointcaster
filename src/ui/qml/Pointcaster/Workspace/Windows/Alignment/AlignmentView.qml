import QtQuick
import QtQuick3D
import QtQuick3D.Helpers
import Pointcaster 1.0
import Pointcaster.Workspace 1.0
import Pointcaster.Geometry 1.0

Item {
    id: root

    required property QtObject deviceAdapter

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

        OrbitViewController {
            id: viewController
            anchors.fill: parent
            camera: cameraNode
            origin: orbitOrigin
        }

        environment: SceneEnvironment {
            clearColor: ThemeColors.shadow
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.NoAA
        }
    }

    MouseArea {
        anchors.fill: parent
        propagateComposedEvents: true
        onClicked: (mouse) => 
            colorPicker.pick(mainView, Math.round(mouse.x), Math.round(mouse.y));
    }

    ColorPicker {
        id: colorPicker
    }

    function snapshot() {
        geo.updateGeometry();
        orbitOrigin.position = geo.boundsCenter;
    }
}
