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
        anchors.fill: parent
        camera: cameraNode

        Model {
            id: pointcloud

            geometry: PointCloudGeometry {
                id: geo
                pointCloudAdapter: root.deviceAdapter ? root.deviceAdapter.pointCloudAdapter() : null
            }

            materials: [
                PointCloudMaterial {}
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
            anchors.fill: parent
            camera: cameraNode
            origin: orbitOrigin
            acceptedButtons: Qt.LeftButton
        }

        environment: SceneEnvironment {
            clearColor: ThemeColors.shadow
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.NoAA
        }
    }

    function snapshot() {
        geo.updateGeometry();
        orbitOrigin.position = geo.boundsCenter;
    }
}
