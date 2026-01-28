import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick3D

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    required property View3D view3d
    required property Node gizmoTarget
    required property Node orbitOrigin

    // camera lock
    readonly property bool viewLocked: lockCameraButton.checked
    readonly property bool orbitRotationRunning: orbitRotationAnim.running

    // guides (grid) toggle
    property bool gridEnabled: true

    property bool originGizmoCollapsed: false
    property bool cameraToolbarCollapsed: false

    readonly property int originGizmoPanelSize: Math.round(92 * Scaling.uiScale)
    readonly property int panelMargin: Math.round(4 * Scaling.uiScale)

    signal requestHomeCamera

    width: originGizmoPanelSize
    implicitWidth: width
    implicitHeight: controlsLayout.implicitHeight

    x: parent ? (parent.width - width) : 0
    y: 0

    onWidthChanged: if (parent)
        x = parent.width - width
    Connections {
        target: root.parent
        function onWidthChanged() {
            root.x = root.parent.width - root.width;
        }
    }

    ColumnLayout {
        id: controlsLayout
        spacing: Math.round(13 * Scaling.uiScale)
        Layout.alignment: Qt.AlignRight | Qt.AlignTop

        Item {
            id: originGizmoCollapser
            width: root.originGizmoPanelSize
            implicitWidth: width

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
                        topRightRadius: 0
                        topLeftRadius: 0
                        bottomRightRadius: 0
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

                    leftPadding: Math.round(3 * Scaling.uiScale)
                    rightPadding: Math.round(3 * Scaling.uiScale)
                    topPadding: 0
                    bottomPadding: 0

                    topLeftRadius: 0
                    topRightRadius: 0

                    backgroundColor: ThemeColors.dark
                    hoverColor: ThemeColors.dark
                    pressedColor: ThemeColors.middark

                    opacity: hovered ? 1 : 0.6
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

            Row {
                id: cameraToolbarRow
                spacing: 0
                y: 0
                x: parent.width - width

                IconButton {
                    id: toggleCameraToolbarButton
                    width: cameraToolbarCollapser.toggleButtonWidth
                    implicitHeight: cameraToolbarCollapser.toggleButtonHeight

                    anchors.top: parent.top
                    anchors.topMargin: Math.round(5 * Scaling.uiScale)

                    iconSource: FontAwesome.icon(root.cameraToolbarCollapsed ? "solid/caret-left" : "solid/caret-right")
                    iconSize: Math.round(9 * Scaling.uiScale)

                    leftPadding: 0
                    rightPadding: 0
                    topPadding: Math.round(3 * Scaling.uiScale)
                    bottomPadding: Math.round(3 * Scaling.uiScale)

                    topRightRadius: 0
                    bottomRightRadius: 0

                    backgroundColor: ThemeColors.dark
                    hoverColor: ThemeColors.dark
                    pressedColor: ThemeColors.middark

                    opacity: hovered ? 1 : 0.6
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
                            border.width: 0
                            topLeftRadius: Math.round(5 * Scaling.uiScale)
                            topRightRadius: 0
                            bottomLeftRadius: Math.round(5 * Scaling.uiScale)
                            bottomRightRadius: 0
                        }

                        contentItem: ColumnLayout {
                            spacing: Math.round(6 * Scaling.uiScale)

                            ToolButton {
                                id: lockCameraButton

                                Layout.preferredWidth: Math.round(24 * Scaling.uiScale)
                                Layout.preferredHeight: Math.round(24 * Scaling.uiScale)

                                checkable: true
                                checked: false

                                contentItem: Image {
                                    source: FontAwesome.icon(lockCameraButton.checked ? "solid/lock" : "solid/lock-open")
                                    width: lockCameraButton.checked ? Math.round(12 * Scaling.uiScale) : Math.round(14 * Scaling.uiScale)
                                    height: width
                                    anchors.centerIn: parent
                                    smooth: true
                                    mipmap: true
                                }

                                InfoToolTip {
                                    visible: parent.hovered
                                    textValue: lockCameraButton.checked ? "Unlock camera view" : "Lock camera view"
                                    delay: 700
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

                                InfoToolTip {
                                    visible: parent.hovered && parent.enabled
                                    textValue: "Re-home camera view"
                                    delay: 700
                                }

                                onClicked: {
                                    if (root.viewLocked)
                                        return;
                                    root.requestHomeCamera();
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

                                contentItem: Image {
                                    source: FontAwesome.icon("solid/table-cells-large")
                                    width: Math.round(16 * Scaling.uiScale)
                                    height: width
                                    anchors.centerIn: parent
                                    smooth: true
                                    mipmap: true
                                }

                                onToggled: root.gridEnabled = checked

                                InfoToolTip {
                                    visible: parent.hovered && parent.enabled
                                    textValue: guidesButton.checked ? "Disable grid" : "Enable grid"
                                    delay: 700
                                }
                            }
                        }
                    }
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
    }
}
