import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick3D

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    required property var workspace

    required property View3D view3d
    required property Node gizmoTarget
    required property Node orbitOrigin
    required property Item sessionView

    readonly property bool orbitRotationRunning: orbitRotationAnim.running

    property bool viewLocked: false
    property bool gridEnabled: true
    property bool orthographicEnabled: false

    property bool originGizmoCollapsed: false
    property bool cameraToolbarCollapsed: false

    property bool consolePanelCollapsed: true
    property bool consolePanelWidthExpanded: false

    readonly property int originGizmoPanelSize: Math.round(92 * Scaling.uiScale)
    readonly property int panelMargin: Math.round(4 * Scaling.uiScale)

    readonly property int consolePanelHeight: Math.round(200 * Scaling.uiScale)

    property int consolePanelWidth: Math.round(500 * Scaling.uiScale)
    readonly property int consolePanelWidthMin: Math.round(200 * Scaling.uiScale)

    signal requestHomeCamera

    anchors.fill: parent

    ColumnLayout {
        id: cameraControlsLayout
        spacing: Math.round(13 * Scaling.uiScale)
        Layout.alignment: Qt.AlignRight | Qt.AlignTop

        width: parent.width
        height: parent.height

        Item {
            id: originGizmoCollapser
            width: root.originGizmoPanelSize
            implicitWidth: width
            Layout.alignment: Qt.AlignRight | Qt.AlignTop

            implicitHeight: originGizmoPanel.height + toggleOriginGizmoButton.implicitHeight
            height: implicitHeight

            Column {
                id: originGizmoColumn
                spacing: 0
                width: parent.width

                Item {
                    id: originGizmoPanel
                    width: parent.width
                    clip: true

                    height: root.originGizmoCollapsed ? 0 : root.originGizmoPanelSize

                    Behavior on height {
                        NumberAnimation {
                            duration: 160
                            easing.type: Easing.OutCubic
                        }
                    }

                    Rectangle {
                        anchors.fill: parent
                        color: ThemeColors.dark
                        opacity: 0.75
                        bottomLeftRadius: Math.round(7 * Scaling.uiScale)
                    }

                    OriginGizmo {
                        id: originGizmo
                        targetNode: root.gizmoTarget

                        enabled: !root.viewLocked
                        opacity: enabled ? 1.0 : 0.35

                        width: root.originGizmoPanelSize - 2 * root.panelMargin
                        height: width
                        anchors.centerIn: parent

                        function removeGizmoBasis(q) {
                            return root.view3d.gizmoBasisInv.times(q).times(root.view3d.gizmoBasis);
                        }
                        function applyGizmoBasis(q) {
                            return root.view3d.gizmoBasis.times(q).times(root.view3d.gizmoBasisInv);
                        }

                        function remapAxisForCurrentFrame(axis) {
                            switch (axis) {
                            case OriginGizmo.Axis.PositiveX:
                                return OriginGizmo.Axis.NegativeX;
                            case OriginGizmo.Axis.NegativeX:
                                return OriginGizmo.Axis.PositiveX;
                            case OriginGizmo.Axis.PositiveY:
                                return OriginGizmo.Axis.NegativeY;
                            case OriginGizmo.Axis.NegativeY:
                                return OriginGizmo.Axis.PositiveY;
                            default:
                                return axis;
                            }
                        }

                        onAxisClicked: axis => {
                            if (!enabled)
                                return;

                            const axisFixed = remapAxisForCurrentFrame(axis);

                            const rotUnbased = removeGizmoBasis(root.gizmoTarget.rotation);
                            const snappedUnbased = originGizmo.quaternionForAxis(axisFixed, rotUnbased);
                            const snappedBased = applyGizmoBasis(snappedUnbased);

                            orbitRotationAnim.to = root.view3d.gizmoToOrbitRotation(snappedBased);
                            orbitRotationAnim.start();
                        }

                        onBallMoved: velocity => {
                            if (!enabled)
                                return;

                            const v = velocity.x;
                            if (Math.abs(v) < 1)
                                return;

                            const rotUnbased = removeGizmoBasis(root.gizmoTarget.rotation);
                            const nextUnbased = (v >= 0) ? originGizmo.quaternionRotateRight(rotUnbased) : originGizmo.quaternionRotateLeft(rotUnbased);

                            const nextBased = applyGizmoBasis(nextUnbased);
                            orbitRotationAnim.to = root.view3d.gizmoToOrbitRotation(nextBased);
                            orbitRotationAnim.start();
                        }
                    }
                }

                IconButton {
                    id: toggleOriginGizmoButton
                    width: Math.round(16 * Scaling.uiScale)
                    implicitHeight: Math.round(11 * Scaling.uiScale)

                    anchors.right: parent.right
                    anchors.rightMargin: root.panelMargin

                    iconSource: FontAwesome.icon(root.originGizmoCollapsed ? "solid/caret-down" : "solid/caret-up")
                    iconSize: Math.round(9 * Scaling.uiScale)
                    iconColor: !pressed ? ThemeColors.mid : ThemeColors.midlight
                    opacity: 0.75

                    topLeftRadius: 0
                    topRightRadius: 0

                    leftPadding: Math.round(3 * Scaling.uiScale)
                    rightPadding: Math.round(3 * Scaling.uiScale)

                    backgroundColor: ThemeColors.dark
                    hoverColor: ThemeColors.middark
                    pressedColor: ThemeColors.mid

                    borderWidth: 0
                    onClicked: root.originGizmoCollapsed = !root.originGizmoCollapsed
                }
            }
        }

        Item {
            id: cameraToolbarCollapser

            readonly property int toggleButtonWidth: Math.round(11 * Scaling.uiScale)
            readonly property int toggleButtonHeight: Math.round(16 * Scaling.uiScale)

            Layout.fillWidth: true
            implicitHeight: Math.max(toggleCameraToolbarButton.implicitHeight, cameraToolbar.implicitHeight)

            Column {
                id: cameraToolbarRow
                spacing: 0
                x: parent.width - width

                IconButton {
                    id: toggleCameraToolbarButton
                    width: cameraToolbarCollapser.toggleButtonWidth
                    implicitHeight: cameraToolbarCollapser.toggleButtonHeight

                    anchors.right: parent.right

                    iconSource: FontAwesome.icon(root.cameraToolbarCollapsed ? "solid/caret-left" : "solid/caret-right")
                    iconSize: Math.round(9 * Scaling.uiScale)
                    iconColor: !pressed ? ThemeColors.mid : ThemeColors.midlight
                    opacity: 0.75

                    topRightRadius: 0
                    bottomRightRadius: 0

                    topPadding: Math.round(3 * Scaling.uiScale)
                    bottomPadding: Math.round(3 * Scaling.uiScale)

                    backgroundColor: ThemeColors.dark
                    hoverColor: ThemeColors.middark
                    pressedColor: ThemeColors.mid

                    borderWidth: 0
                    onClicked: root.cameraToolbarCollapsed = !root.cameraToolbarCollapsed
                }

                Item {
                    id: cameraToolbarPanel
                    height: cameraToolbar.implicitHeight
                    clip: true

                    width: root.cameraToolbarCollapsed ? 0 : cameraToolbar.implicitWidth

                    Behavior on width {
                        NumberAnimation {
                            duration: 160
                            easing.type: Easing.OutCubic
                        }
                    }

                    ToolBar {
                        id: cameraToolbar
                        anchors.fill: parent
                        padding: 0
                        topPadding: Math.round(5 * Scaling.uiScale)
                        bottomPadding: Math.round(5 * Scaling.uiScale)

                        background: Rectangle {
                            color: ThemeColors.dark
                            opacity: 0.75
                            topLeftRadius: Math.round(5 * Scaling.uiScale)
                            bottomLeftRadius: Math.round(5 * Scaling.uiScale)
                        }

                        contentItem: ColumnLayout {
                            spacing: Math.round(6 * Scaling.uiScale)

                            ToolButton {
                                id: lockCameraButton
                                Layout.preferredWidth: Math.round(24 * Scaling.uiScale)
                                Layout.preferredHeight: Math.round(24 * Scaling.uiScale)

                                checkable: true
                                checked: root.viewLocked
                                onToggled: root.viewLocked = checked

                                contentItem: Image {
                                    source: FontAwesome.icon(lockCameraButton.checked ? "solid/lock" : "solid/lock-open")
                                    width: lockCameraButton.checked ? Math.round(12 * Scaling.uiScale) : Math.round(14 * Scaling.uiScale)
                                    height: width
                                    anchors.centerIn: parent
                                    smooth: true
                                    mipmap: true
                                }

                                InfoToolTip {
                                    delay: 800
                                    textValue: lockCameraButton.checked ? "Unlock camera" : "Lock camera"
                                }
                            }

                            ToolButton {
                                id: homeCameraButton
                                enabled: !root.viewLocked
                                opacity: enabled ? 1.0 : 0.35

                                Layout.preferredWidth: Math.round(24 * Scaling.uiScale)
                                Layout.preferredHeight: Math.round(24 * Scaling.uiScale)

                                icon.source: FontAwesome.icon("solid/house")
                                icon.width: Math.round(18 * Scaling.uiScale)
                                icon.height: Math.round(18 * Scaling.uiScale)

                                onClicked: root.requestHomeCamera()

                                InfoToolTip {
                                    delay: 800
                                    textValue: "Home orientation"
                                }
                            }

                            ToolButton {
                                id: guidesButton
                                enabled: !root.viewLocked
                                opacity: enabled ? 1.0 : 0.35

                                Layout.preferredWidth: Math.round(24 * Scaling.uiScale)
                                Layout.preferredHeight: Math.round(24 * Scaling.uiScale)

                                checkable: true
                                checked: root.gridEnabled
                                onToggled: root.gridEnabled = checked

                                contentItem: Image {
                                    source: FontAwesome.icon("solid/table-cells-large")
                                    width: Math.round(16 * Scaling.uiScale)
                                    height: width
                                    anchors.centerIn: parent
                                    smooth: true
                                    mipmap: true
                                }

                                InfoToolTip {
                                    delay: 800
                                    textValue: guidesButton.checked ? "Hide floor plane" : "Show floor plane"
                                }
                            }

                            ToolButton {
                                id: projectionButton
                                enabled: !root.viewLocked
                                opacity: enabled ? 1.0 : 0.35

                                Layout.preferredWidth: Math.round(24 * Scaling.uiScale)
                                Layout.preferredHeight: Math.round(24 * Scaling.uiScale)

                                checkable: true
                                checked: root.orthographicEnabled
                                onToggled: root.orthographicEnabled = checked

                                contentItem: Image {
                                    source: FontAwesome.icon("solid/cube")
                                    width: Math.round(16 * Scaling.uiScale)
                                    height: width
                                    anchors.centerIn: parent
                                    smooth: true
                                    mipmap: true
                                }

                                InfoToolTip {
                                    delay: 800
                                    textValue: projectionButton.checked ? "Disable orthographic projection" : "Enable orthographic projection"
                                }
                            }
                        }
                    }
                }
            }
        }

        Item {
            id: workspaceControls
            Layout.fillWidth: true
            Layout.fillHeight: true

            readonly property int consoleRowHeight: 20 * Scaling.uiScale

            Component {
                id: consoleRow

                ItemDelegate {
                    required property var modelData

                    width: root.consolePanelWidth
                    height: workspaceControls.consoleRowHeight

                    background: Rectangle {
                        color: "transparent"
                    }

                    contentItem: Row {
                        height: parent.height
                        spacing: 0

                        Text {
                            text: "["
                            font: Scaling.monoFont
                            color: ThemeColors.text
                        }

                        Text {
                            text: modelData.logLevel
                            font: Scaling.monoFont
                            color: ThemeColors[modelData.logLevelColor]
                        }

                        Text {
                            text: "]"
                            font: Scaling.monoFont
                            color: ThemeColors.text
                        }

                        Text {
                            text: modelData.message
                            leftPadding: 6 * Scaling.uiScale
                            font: Scaling.monoFont
                            color: ThemeColors.text
                        }
                    }
                }
            }

            Column {
                id: consoleColumn
                height: toggleConsoleButton.height + consolePanel.height
                width: root.consolePanelWidthExpanded ? root.width : root.consolePanelWidth

                x: parent.width - width
                y: parent.height - height

                IconButton {
                    id: toggleConsoleButton
                    z: 99
                    width: Math.round(16 * Scaling.uiScale)
                    implicitHeight: Math.round(11 * Scaling.uiScale)

                    anchors.right: parent.right
                    anchors.rightMargin: root.panelMargin

                    iconSource: FontAwesome.icon(root.consolePanelCollapsed ? "solid/caret-up" : "solid/caret-down")
                    iconSize: Math.round(9 * Scaling.uiScale)
                    iconColor: !pressed ? ThemeColors.mid : ThemeColors.midlight
                    opacity: 0.75

                    leftPadding: Math.round(3 * Scaling.uiScale)
                    rightPadding: Math.round(3 * Scaling.uiScale)

                    bottomLeftRadius: 0
                    bottomRightRadius: 0

                    backgroundColor: ThemeColors.dark
                    hoverColor: ThemeColors.middark
                    pressedColor: ThemeColors.mid

                    borderWidth: 0
                    onClicked: root.consolePanelCollapsed = !root.consolePanelCollapsed
                }

                Item {
                    id: consolePanel

                    width: parent.width
                    height: root.consolePanelCollapsed ? 0 : consolePanelHeight

                    opacity: root.consolePanelCollapsed ? 0.66 : 1

                    Behavior on height {
                        NumberAnimation {
                            duration: 160
                            easing.type: Easing.OutCubic
                        }
                    }

                    Rectangle {
                        color: ThemeColors.dark
                        opacity: 0.75
                        topLeftRadius: root.consolePanelWidthExpanded ? 0 : Math.round(7 * Scaling.uiScale)
                        width: parent.width
                        height: parent.height
                    }

                    ScrollView {
                        id: consolePanelScrollView
                        width: parent.width
                        height: parent.height
                        clip: true

                        wheelEnabled: !root.consolePanelCollapsed
                        ScrollBar.vertical.policy: root.consolePanelCollapsed ? ScrollBar.AlwaysOff : ScrollBar.AlwaysOn

                        ListView {
                            boundsBehavior: Flickable.StopAtBounds
                            spacing: 0

                            model: root.workspace ? root.workspace.consoleHistoryEntries : []
                            delegate: consoleRow
                        }
                    }

                    Connections {
                        target: workspace
                        function onConsoleHistoryEntriesChanged() {
                            // scroll to bottom when new console entries added
                            consolePanelScrollView.ScrollBar.vertical.position = consolePanelScrollView.contentHeight;
                        }
                    }
                }
            }

            Item {
                id: consoleOverlay
                visible: consoleColumn.height == toggleConsoleButton.height

                height: consoleOverlayColumn.height
                width: root.consolePanelWidth

                x: parent.width - width
                y: parent.height - height - toggleConsoleButton.height

                Rectangle {
                    color: ThemeColors.dark
                    opacity: 0.35
                    width: parent.width
                    height: parent.height
                }

                Column {
                    id: consoleOverlayColumn
                    spacing: 0

                    Repeater {
                        model: root.workspace ? root.workspace.consoleOverlayEntries : []
                        delegate: consoleRow
                    }
                }
            }

            Item {
                id: consoleResizeBar
                y: consoleColumn.y + toggleConsoleButton.height
                height: consolePanel.height

                width: Math.round(10 * Scaling.uiScale)
                x: consoleColumn.x - Math.round(width / 2)

                Rectangle {
                    width: parent.width
                    height: parent.height
                    color: "transparent"

                    MouseArea {
                        id: consoleResizeDrag
                        z: 50
                        width: parent.width
                        height: parent.height
                        hoverEnabled: true
                        acceptedButtons: Qt.LeftButton
                        cursorShape: Qt.SplitHCursor
                        property bool dragging: false
                        onPressed: dragging = true
                        onReleased: dragging = false
                        onCanceled: dragging = false
                        preventStealing: true
                        onPositionChanged: {
                            if (!dragging)
                                return;
                            let newWidth = Math.round(root.consolePanelWidth - mouseX);
                            newWidth = Math.max(newWidth, root.consolePanelWidthMin);
                            newWidth = Math.min(newWidth, root.width);
                            root.consolePanelWidth = newWidth;
                            root.consolePanelWidthExpanded = newWidth >= root.width - consoleResizeBar.width;
                        }
                    }
                }

                Connections {
                    target: root
                    function onWidthChanged() {
                        if (root.consolePanelWidthExpanded) {
                            root.consolePanelWidth = root.width
                        }
                    }
                }
            }

            Timer {
                // update the console every 100ms
                interval: 100
                running: true
                repeat: true
                onTriggered: {
                    root.workspace.syncConsole();
                    // console.log(root.workspace.consoleOverlayEntries())
                }
            }
        }
    }

    PropertyAnimation {
        id: orbitRotationAnim
        target: root.orbitOrigin
        property: "rotation"
        duration: 220
        easing.type: Easing.OutCubic
        onStopped: {
            if (root.sessionView == null || root.sessionView == undefined)
                return;
            root.sessionView.commitCameraTransformToConfig();
        }
    }
}
