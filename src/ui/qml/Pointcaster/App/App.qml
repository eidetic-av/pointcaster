import QtQuick
import QtQuick.Controls.Fusion
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick3D
import QtQuick3D.Helpers
import Pointcaster.Controls
import com.kdab.dockwidgets as KDDW

ApplicationWindow {
    id: root
    visible: true
    title: "Pointcaster"

    minimumWidth: 1200
    minimumHeight: 800

    palette {
        window:            "#1e1e2e"
        windowText:        "#cdd6f4"
        base:              "#181825"
        text:              "#cdd6f4"
        button:            "#313244"
        buttonText:        "#cdd6f4"
        highlight:         "#89b4fa"
        highlightedText:   "#1e1e2e"
    }

    FontLoader {
        id: regularFont
        source: "qrc:/fonts/AtkinsonHyperlegibleNext-Regular.otf"
    }

    font.family: regularFont.name

    menuBar : MenuBar {

        Menu {
            title: qsTr("&File")

            Action { text: qsTr("&Save layout") }
            Action { text: qsTr("&Restore layout") }

            MenuSeparator{}
            Action {
                text: qsTr("&Quit")
                onTriggered: Qt.quit()
            }
        }

        Menu {
            title: qsTr("&Window")
        }

        Menu {
            title: qsTr("&Help")

            Action {
                text: qsTr("&About")
                onTriggered: aboutPopup.open()
            }
        }
    }

    KDDW.DockingArea {
        id: dockingArea
        anchors.fill: parent
        uniqueName: "MainDockingArea"

        KDDW.DockWidget {
            id: viewWindow
            uniqueName: "viewWindow"
            title: "3D View"

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

                Node {
                    id: orbitOrigin
                    PerspectiveCamera { id: camera; z: 500 }
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

                OrbitCameraController {
                    anchors.fill: parent
                    camera: camera
                    origin: orbitOrigin
                    acceptedButtons: Qt.LeftButton
                    xSpeed: 0.5
                    ySpeed: 0.5
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
        }

        KDDW.DockWidget {
            id: debugViewWindow
            uniqueName: "debugViewWindow"
            title: "Debug"

            Pane {
                id: debugPane
                anchors.fill: parent
                // padding: 8
                DebugView {
                    id: debugView
                    anchors.fill: parent
                    // anchors.margins: 4
                    source: view
                    resourceDetailsVisible: true
                    // color: root.palette.base
                }
            }
        }


        Component.onCompleted: {
            addDockWidget(viewWindow, KDDW.KDDockWidgets.Location_OnLeft)
            addDockWidget(debugViewWindow,
                          KDDW.KDDockWidgets.Location_OnRight,
                          viewWindow)
        }
    }

    Popup {
        id: aboutPopup
        modal: true
        dim: true
        focus: true
        width: 360
        height: 200

        x: (parent.width - width) / 2
        y: (parent.height - height) / 2

        background: Rectangle {
            color: "#181825"
            radius: 8
            border.color: "#89b4fa"
        }

        Column {
            anchors.centerIn: parent
            spacing: 12

            Label {
                text: "Pointcaster\n\nAbout dialog placeholder"
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
            }

            Button {
                text: "Close"
                onClicked: aboutPopup.close()
            }
        }
    }

}

