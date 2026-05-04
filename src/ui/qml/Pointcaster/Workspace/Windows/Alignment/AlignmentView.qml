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
        onClicked: mouse => {
            // on click, we determine the color of the pixel clicked
            const x = Math.round(mouse.x);
            const y = Math.round(mouse.y);
            colorPicker.run(x, y);
            // mainView.grabToImage(function (result) {
            //     console.log(`generated image...`);
            //     console.log(`clicked at: ${x}, ${y}`);
            //     console.log(result.image.value() == null)
            //     console.log(result.image.value == null)
            //     // console.log(result.image);
            //     // const color = result.image.value().pixel(x, y);
            //     // console.log(color);
            // });
            // mouse.accepted = false;
        }
    }

    ColorPicker {
        id: colorPicker
    }

    function snapshot() {
        geo.updateGeometry();
        orbitOrigin.position = geo.boundsCenter;
    }
}
