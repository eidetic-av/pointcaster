import QtQuick
import QtQuick.Controls.Fusion
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import QtQuick3D
import QtQuick3D.Helpers
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

ApplicationWindow {
    id: root
    visible: true
    title: "Pointcaster"

    minimumWidth: 1200
    minimumHeight: 800

    // palette {
    //     window: "#1e1e2e"
    //     windowText: "#cdd6f4"
    //     base: "#181825"
    //     text: "#cdd6f4"
    //     button: "#313244"
    //     buttonText: "#cdd6f4"
    //     highlight: "#89b4fa"
    //     highlightedText: "#1e1e2e"
    // }

    FontLoader {
        id: regularFont
        source: "../Resources/Fonts/AtkinsonHyperlegibleNext-Regular.otf"
    }

    font.family: regularFont.name

    FileDialog {
        id: openWorkspaceDialog
        nameFilters: [
            qsTr("JSON files (*.json)"),
            qsTr("All files (*)")
        ]
        onAccepted: workspaceModel.loadFromFile(selectedFile)
    }

    menuBar : MenuBar {
        Menu {
            title: qsTr("&File")

            Action { text: qsTr("&New Workspace") }
            Action { text: qsTr("New &Session") }

            MenuSeparator {}

            Action {
                text: qsTr("&Open")
                onTriggered: openWorkspaceDialog.open()
            }

            MenuSeparator {}

            Action {
                text: qsTr("&Save")
                onTriggered: workspaceModel.save()
            }

            MenuSeparator {}

            Action {
                text: qsTr("&Quit")
                onTriggered: workspaceModel.close()
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

        KDDW.DockWidget {
            id: devicesWindow
            uniqueName: "devicesWindow"
            title: "Devices"

            Item {
                id: devices
                anchors.fill: parent
                clip: true

                // Shared state for the properties pane
                property var currentAdapter: configList.currentItem ? configList.currentItem.modelData : null

                    // Top: device selector list
                    ListView {
                        id: configList
                        model: workspaceModel ? workspaceModel.deviceConfigAdapters: []

                        anchors {
                            top: parent.top
                            left: parent.left
                            right: parent.right
                        }

                        // Fixed height "selector" strip
                        height: 160
                        clip: true

                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded
                        }

                        // Remove kinetic "flick" behaviour
                        boundsBehavior: Flickable.StopAtBounds
                        maximumFlickVelocity: 0
                        flickDeceleration: 1000000

                        delegate: Rectangle {
                            id: deviceRow
                            required property int index
                            required property ConfigAdapter modelData

                            width: ListView.view.width
                            height: 40

                            color: ListView.isCurrentItem
                            ? Qt.rgba(0.2, 0.4, 0.8, 0.25)
                            : "transparent"

                            border.color: "#404040"
                            border.width: 0.5

                            Row {
                                anchors.fill: parent
                                anchors.leftMargin: 8
                                anchors.rightMargin: 8
                                spacing: 8

                                Text {
                                    text: modelData.displayName()
                                    verticalAlignment: Text.AlignVCenter
                                    elide: Text.ElideRight
                                }

                                Text {
                                    text: modelData.fieldValue(1) ? "active" : "inactive"
                                    verticalAlignment: Text.AlignVCenter
                                    color: modelData.fieldValue(1) ? "#6fb46f" : "#b46f6f"
                                }
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: configList.currentIndex = deviceRow.index
                            }
                        }
                    }

                    // inspector for selected device
                    Item {
                        id: propertiesPane
                        anchors {
                            top: configList.bottom
                            left: parent.left
                            right: parent.right
                            bottom: parent.bottom
                        }
                        anchors.topMargin: 6

                        // Shared width of the left "label" column
                        property int labelColumnWidth: width * 0.35

                            // Title / empty state
                            Text {
                                id: header
                                anchors {
                                    top: parent.top
                                    left: parent.left
                                    right: parent.right
                                    leftMargin: 8
                                    rightMargin: 8
                                    topMargin: 4
                                }
                                text: devices.currentAdapter
                                ? "Device properties — " + devices.currentAdapter.displayName()
                                : "No device selected"
                                font.bold: true
                                elide: Text.ElideRight
                            }

                            Flickable {
                                id: propertiesFlick
                                anchors {
                                    top: header.bottom
                                    topMargin: 8
                                    left: parent.left
                                    right: parent.right
                                    bottom: parent.bottom
                                    leftMargin: 8
                                    rightMargin: 8
                                    bottomMargin: 8
                                }

                                contentWidth: width
                                contentHeight: propertiesColumn.implicitHeight
                                clip: true

                                ScrollBar.vertical: ScrollBar {
                                    policy: ScrollBar.AsNeeded
                                }

                                // Property rows
                                Column {
                                    id: propertiesColumn
                                    width: propertiesFlick.width
                                    spacing: 4

                                    Repeater {
                                        model: devices.currentAdapter ? devices.currentAdapter.fieldCount() : 0

                                        delegate: Item {
                                            required property int index

                                            width: propertiesColumn.width
                                            height: Math.max(nameText.implicitHeight, valueField.implicitHeight) + 6

                                            visible: !devices.currentAdapter.isHidden(index)

                                            // Label region (left column)
                                            Item {
                                                id: labelContainer
                                                anchors {
                                                    left: parent.left
                                                    top: parent.top
                                                    bottom: parent.bottom
                                                }
                                                width: propertiesPane.labelColumnWidth

                                                Text {
                                                    id: nameText
                                                    anchors {
                                                        left: parent.left
                                                        right: parent.right
                                                        verticalCenter: parent.verticalCenter
                                                    }
                                                    horizontalAlignment: Text.AlignRight
                                                    elide: Text.ElideRight
                                                    text: devices.currentAdapter.fieldName(index)
                                                }
                                            }

                                            // value region (right column)
                                            TextField {
                                                id: valueField
                                                anchors {
                                                    left: labelContainer.right
                                                    right: parent.right
                                                    verticalCenter: parent.verticalCenter
                                                }
                                                text: devices.currentAdapter.fieldValue(index)
                                                enabled: !devices.currentAdapter.isDisabled(index)

                                                onEditingFinished: {
                                                    devices.currentAdapter.setFieldValue(index, text)
                                                }
                                            }
                                        }
                                    }
                                }

                                // vertical divider handle
                                Rectangle {
                                    id: dividerHandle
                                    x: propertiesPane.labelColumnWidth - width / 2
                                    width: 4
                                    anchors {
                                        top: parent.top
                                        bottom: parent.bottom
                                    }
                                    color: "#505050"
                                    opacity: 0.6

                                    MouseArea {
                                        anchors.fill: parent
                                        cursorShape: Qt.SplitHCursor
                                        drag.target: dividerHandle
                                        drag.axis: Drag.XAxis
                                        drag.minimumX: 80
                                        drag.maximumX: propertiesPane.width - 80

                                        onPositionChanged: {
                                            propertiesPane.labelColumnWidth = dividerHandle.x + dividerHandle.width / 2
                                        }
                                    }
                                }
                            }
                        }
                    }
                }



                Component.onCompleted: {
                    addDockWidget(viewWindow, KDDW.KDDockWidgets.Location_OnLeft)
                    addDockWidget(debugViewWindow, KDDW.KDDockWidgets.Location_OnRight, viewWindow)
                    addDockWidget(devicesWindow, KDDW.KDDockWidgets.Location_OnLeft)
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

