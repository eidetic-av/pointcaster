import QtQuick
import QtQuick3D
import QtQuick3D.Helpers
import QtQuick.Controls
import QtQuick.Layouts
import Pointcaster 1.0

Item {
    id: root

    required property QtObject primaryAdapter
    required property QtObject secondaryAdapter
    required property QtObject alignmentController

    // 0 = "color" (PointCloudMaterial), >1 = solid color index
    property int primaryColorMode: 0
    property int secondaryColorMode: 1

    readonly property var solidColors: [ThemeColors.greyBlue, ThemeColors.red, ThemeColors.green, ThemeColors.blue, ThemeColors.yellow]
    readonly property var solidColorNames: ["Grey", "Red", "Green", "Blue", "Yellow"]

    View3D {
        id: view
        anchors.fill: parent
        camera: cameraNode

        // primary cloud with regular color
        Model {
            visible: root.primaryColorMode === 0
            geometry: primaryGeo
            materials: [
                PointCloudMaterial {
                    uPointSize: viewController.shaderPointSize
                }
            ]
        }

        // primary cloud with solid color
        Model {
            visible: root.primaryColorMode !== 0
            geometry: primaryGeo
            materials: [
                PointSolidMaterial {
                    uPointSize: viewController.shaderPointSize
                    uColor: root.primaryColorMode > 0 ? root.solidColors[root.primaryColorMode - 1] : "white"
                }
            ]
        }

        PointCloudGeometry {
            id: primaryGeo
            pointCloudAdapter: root.primaryAdapter ? root.primaryAdapter.pointCloudAdapter() : null
        }

        // secondary cloud is wrapped in a node that controls its transformation based on the result of the alignment controller
        Node {
            id: secondaryTransformNode
            position: root.alignmentController.resultPosition
            rotation: root.alignmentController.resultRotation

            // secondary cloud with regular color
            Model {
                visible: root.secondaryColorMode === 0
                geometry: secondaryGeo
                materials: [
                    PointCloudMaterial {
                        uPointSize: viewController.shaderPointSize
                    }
                ]
            }

            // secondary cloud with solid color
            Model {
                visible: root.secondaryColorMode !== 0
                geometry: secondaryGeo
                materials: [
                    PointSolidMaterial {
                        uPointSize: viewController.shaderPointSize
                        uColor: root.secondaryColorMode > 0 ? root.solidColors[root.secondaryColorMode - 1] : "white"
                    }
                ]
            }
        }

        PointCloudGeometry {
            id: secondaryGeo
            pointCloudAdapter: root.secondaryAdapter ? root.secondaryAdapter.pointCloudAdapter() : null
        }

        Node {
            id: orbitOrigin

            PerspectiveCamera {
                id: cameraNode
                z: 300
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
        }
    }

    // material picker
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.margins: Math.round(Scaling.uiScale * 8)
        width: col.implicitWidth + Math.round(Scaling.uiScale * 16)
        height: col.implicitHeight + Math.round(Scaling.uiScale * 16)
        radius: Math.round(Scaling.uiScale * 6)
        color: ThemeColors.withAlpha(ThemeColors.dark, 0.85)

        Column {
            id: col
            anchors.centerIn: parent
            spacing: Math.round(Scaling.uiScale * 8)

            ColorModeSelector {
                label: "Primary"
                currentMode: root.primaryColorMode
                onModeSelected: mode => root.primaryColorMode = mode
            }

            Rectangle {
                width: parent.width
                height: 1
                color: ThemeColors.middark
            }

            ColorModeSelector {
                label: "Secondary"
                currentMode: root.secondaryColorMode
                onModeSelected: mode => root.secondaryColorMode = mode
            }
        }
    }

    component ColorModeSelector: Column {
        property string label: ""
        property int currentMode: 0
        signal modeSelected(int mode)

        spacing: Math.round(Scaling.uiScale * 4)

        Text {
            text: label
            font: Scaling.uiSmallFont
            color: ThemeColors.midlight
        }

        Row {
            spacing: Math.round(Scaling.uiScale * 4)

            // color option (regular shading)
            Rectangle {
                width: Math.round(Scaling.uiScale * 36)
                height: Math.round(Scaling.uiScale * 20)
                radius: Math.round(Scaling.uiScale * 3)
                border.width: 1
                border.color: currentMode === 0 ? ThemeColors.highlight : ThemeColors.middark
                color: "transparent"

                opacity: rgbMouseArea.containsMouse ? 1 : currentMode === 0 ? 0.8 : 0.4

                // mini rainbow gradient to signify regular shading
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop {
                        position: 0.0
                        color: ThemeColors.red
                    }
                    GradientStop {
                        position: 0.5
                        color: ThemeColors.highlight
                    }
                    GradientStop {
                        position: 1.0
                        color: ThemeColors.blue
                    }
                }

                Text {
                    anchors.centerIn: parent
                    text: "RGB"
                    font.pointSize: Scaling.smallPointSize
                    font.weight: Font.Bold
                    color: "white"
                    style: Text.Outline
                    styleColor: Qt.rgba(0, 0, 0, 0.6)
                }

                MouseArea {
                    id: rgbMouseArea
                    anchors.fill: parent
                    onClicked: modeSelected(0)
                    hoverEnabled: true
                }
            }

            // Solid color swatches
            Repeater {
                model: root.solidColors

                Rectangle {
                    width: Math.round(Scaling.uiScale * 20)
                    height: Math.round(Scaling.uiScale * 20)
                    radius: Math.round(Scaling.uiScale * 3)
                    color: modelData
                    border.width: 1
                    border.color: currentMode === index + 1 ? ThemeColors.highlight : ThemeColors.middark
                    opacity: mouseArea.containsMouse ? 1 : currentMode === index + 1 ? 0.8 : 0.4

                    MouseArea {
                        id: mouseArea
                        anchors.fill: parent
                        onClicked: modeSelected(index + 1)
                        hoverEnabled: true
                    }
                }
            }
        }
    }

    function activate() {
        primaryGeo.updateGeometry();
        secondaryGeo.updateGeometry();
        orbitOrigin.position = primaryGeo.boundsCenter;
    }
}
