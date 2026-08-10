import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Shapes
import QtQuick3D

import Pointcaster 1.0

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

    property real controlSpacing: Math.round(16 * Scaling.uiScale)

    property bool gizmoEnabled: true

    signal requestHomeCamera

    anchors.fill: parent

    // camera frames come the selected operator's host...
    // whether that's a device or the session
    readonly property var frameSource: root.workspace ? root.workspace.selectedOperatorFrameSource : null
    readonly property var frameSlots: frameSource ? frameSource.frameSlots : []
    readonly property var frameUrls: frameSource ? frameSource.frameUrls : ({})

    readonly property var selectedOperator: root.workspace ? root.workspace.selectedOperatorAdapter : null
    readonly property var selectedNodeAdapter: selectedOperator ? selectedOperator.configAdapter : (root.sessionView ? (root.sessionView.selectedDeviceAdapter || root.sessionView.selectedGroupAdapter) : null)

    property string selectedNodeName: ""

    function _fieldText(adapter, path) {
        const v = adapter.value(path);
        return (v === undefined || v === null) ? "" : String(v);
    }

    function updateSelectedNodeName() {
        const adapter = root.selectedNodeAdapter;
        if (!adapter) {
            root.selectedNodeName = "";
            return;
        }
        const label = _fieldText(adapter, "label");
        root.selectedNodeName = label.length > 0 ? label : _fieldText(adapter, "id");
    }

    onSelectedNodeAdapterChanged: updateSelectedNodeName()
    Component.onCompleted: updateSelectedNodeName()

    Connections {
        target: root.selectedNodeAdapter
        ignoreUnknownSignals: true
        function onFieldChanged(path) {
            if (path === "label" || path === "id")
                root.updateSelectedNodeName();
        }
    }

    Item {
        id: sessionControlsOverlay

        width: parent.width
        height: parent.height

        SessionControlCollapser {
            id: originGizmoCollapser

            direction: SessionControlCollapser.CollapseUp

            anchors.right: parent.right
            anchors.top: parent.top

            collapsed: root.originGizmoCollapsed
            onCollapsedChanged: root.originGizmoCollapsed = collapsed

            contentItem: OriginGizmo {
                id: originGizmo
                targetNode: root.gizmoTarget

                enabled: !root.viewLocked
                opacity: enabled ? 1.0 : 0.35

                width: root.originGizmoPanelSize
                height: width

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

        SessionControlCollapser {
            id: cameraToolbarCollapser

            direction: SessionControlCollapser.CollapseRight

            collapsed: root.cameraToolbarCollapsed
            onCollapsedChanged: root.cameraToolbarCollapsed = collapsed

            anchors.right: parent.right
            anchors.top: originGizmoCollapser.bottom
            anchors.topMargin: root.controlSpacing

            contentItem: Column {
                spacing: Math.round(6 * Scaling.uiScale)

                ToolButton {
                    id: lockCameraButton
                    width: Math.round(24 * Scaling.uiScale)
                    height: Math.round(24 * Scaling.uiScale)

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

                    width: Math.round(24 * Scaling.uiScale)
                    height: Math.round(24 * Scaling.uiScale)

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

                    width: Math.round(24 * Scaling.uiScale)
                    height: Math.round(24 * Scaling.uiScale)

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

                    width: Math.round(24 * Scaling.uiScale)
                    height: Math.round(24 * Scaling.uiScale)

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

                ToolButton {
                    id: gizmoButton
                    enabled: !root.viewLocked
                    opacity: enabled ? 1.0 : 0.35

                    width: Math.round(24 * Scaling.uiScale)
                    height: Math.round(24 * Scaling.uiScale)

                    checkable: true
                    checked: root.gizmoEnabled
                    onToggled: root.gizmoEnabled = checked

                    contentItem: Image {
                        source: FontAwesome.icon("solid/up-down-left-right")
                        width: Math.round(16 * Scaling.uiScale)
                        height: width
                        anchors.centerIn: parent
                        smooth: true
                        mipmap: true
                    }

                    InfoToolTip {
                        delay: 800
                        textValue: gizmoButton.checked ? "Hide selection gizmo" : "Show selection gizmo"
                    }
                }
            }
        }

        SessionControlCollapser {
            id: cameraFrameCollapser
            visible: root.frameSlots.length > 0
            direction: SessionControlCollapser.CollapseUp
            buttonAlignment: Qt.AlignLeft
            anchors.top: parent.top
            anchors.left: parent.left

            contentItem: Column {
                spacing: Math.round(4 * Scaling.uiScale)

                // names whatever the camera view below belongs to
                Item {
                    id: selectionHeader
                    readonly property real padding: Math.round(3 * Scaling.uiScale)
                    implicitWidth: headerRow.implicitWidth + padding * 2
                    implicitHeight: headerRow.implicitHeight + padding * 2

                    Row {
                        id: headerRow
                        anchors.centerIn: parent
                        spacing: Math.round(6 * Scaling.uiScale)

                        Image {
                            anchors.verticalCenter: parent.verticalCenter
                            source: FontAwesome.icon("solid/video")
                            width: Math.round(10 * Scaling.uiScale)
                            height: width
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                            mipmap: true
                        }

                        Text {
                            id: selectionName
                            anchors.verticalCenter: parent.verticalCenter
                            text: root.selectedNodeName
                            font: Scaling.uiFont
                            color: ThemeColors.text
                            elide: Text.ElideRight
                            // long ids elide rather than stretching the panel
                            width: Math.min(implicitWidth, Math.round(200 * Scaling.uiScale))
                        }
                    }
                }

                Item {
                    id: frameContainer
                    width: 200
                    height: 200

                    readonly property real navSpacing: Math.round(5 * Scaling.uiScale)

                    StackLayout {
                        id: frameStack
                        currentIndex: 0
                        anchors.fill: parent
                        // the nav row sits in the space this leaves at the bottom
                        anchors.bottomMargin: navRow.height + frameContainer.navSpacing

                        Repeater {
                            model: root.frameSlots
                            Image {
                                // modelData is the channel name string
                                source: root.frameUrls[modelData] || ""
                                cache: false
                                fillMode: Image.PreserveAspectFit
                            }
                        }

                        property string selectedName: ""

                        function resolveIndex() {
                            if (selectedName === "" || root.frameSlots.length === 0)
                                return 0;
                            for (var i = 0; i < root.frameSlots.length; ++i) {
                                if (root.frameSlots[i] === selectedName)
                                    return i;
                            }
                            return 0;
                        }

                        onSelectedNameChanged: currentIndex = resolveIndex()
                    }

                    Connections {
                        target: root
                        function onFrameSlotsChanged() {
                            Qt.callLater(function () {
                                if (frameStack.count > 0)
                                    frameStack.currentIndex = frameStack.resolveIndex();
                            });
                        }
                    }

                    Row {
                        id: navRow
                        anchors.bottom: parent.bottom
                        anchors.horizontalCenter: parent.horizontalCenter
                        spacing: 0

                        IconButton {
                            id: prevBtn
                            enabled: frameStack.count > 1
                            tooltip: "Previous frame"
                            iconSource: FontAwesome.icon("solid/chevron-left")
                            iconSize: Math.round(10 * Scaling.uiScale)
                            topPadding: Math.round(3 * Scaling.uiScale)
                            bottomPadding: Math.round(3 * Scaling.uiScale)
                            leftPadding: Math.round(3 * Scaling.uiScale)
                            rightPadding: Math.round(3 * Scaling.uiScale)
                            onClicked: {
                                var idx = (frameStack.currentIndex - 1 + frameStack.count) % frameStack.count;
                                frameStack.selectedName = root.frameSlots[idx];
                            }
                        }

                        Label {
                            text: frameStack.count > 0 ? "%1 (%2/%3)".arg(root.frameSlots[frameStack.currentIndex]).arg(frameStack.currentIndex + 1).arg(frameStack.count) : ""
                            font: Scaling.uiFont
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                            width: frameContainer.width - prevBtn.width - nextBtn.width
                            topPadding: Math.round(1 * Scaling.uiScale)
                        }

                        IconButton {
                            id: nextBtn
                            enabled: frameStack.count > 1
                            tooltip: "Next frame"
                            iconSource: FontAwesome.icon("solid/chevron-right")
                            iconSize: Math.round(10 * Scaling.uiScale)
                            topPadding: Math.round(3 * Scaling.uiScale)
                            bottomPadding: Math.round(3 * Scaling.uiScale)
                            leftPadding: Math.round(3 * Scaling.uiScale)
                            rightPadding: Math.round(3 * Scaling.uiScale)
                            onClicked: {
                                var idx = (frameStack.currentIndex + 1) % frameStack.count;
                                frameStack.selectedName = root.frameSlots[idx];
                            }
                        }
                    }
                }
            }

            Shape {
                id: resizeHandle
                parent: cameraFrameCollapser.container
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                property real size: Math.round(10 * Scaling.uiScale)
                property real startWidth
                property real startHeight
                width: size
                height: size
                ShapePath {
                    fillColor: resizeHover.hovered ? ThemeColors.midlight : ThemeColors.mid
                    strokeColor: "transparent"
                    startX: resizeHandle.size
                    startY: 0
                    PathLine {
                        x: resizeHandle.size
                        y: resizeHandle.size
                    }
                    PathLine {
                        x: 0
                        y: resizeHandle.size
                    }
                    PathLine {
                        x: resizeHandle.size
                        y: 0
                    }
                }
                HoverHandler {
                    id: resizeHover
                    cursorShape: Qt.SizeFDiagCursor
                }
                DragHandler {
                    id: resizeDrag
                    target: null
                    onActiveChanged: {
                        if (!active)
                            return;

                        resizeHandle.startWidth = frameContainer.width;
                        resizeHandle.startHeight = frameContainer.height;
                    }
                    onActiveTranslationChanged: {
                        frameContainer.width = Math.max(40, resizeHandle.startWidth + activeTranslation.x);
                        frameContainer.height = Math.max(40, resizeHandle.startHeight + activeTranslation.y);
                    }
                }
            }
        }

        Item {
            id: workspaceControls
            anchors.fill: parent

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

                    PaddedScrollView {
                        id: consolePanelScrollView
                        width: parent.width
                        height: parent.height
                        clip: true

                        wheelEnabled: !root.consolePanelCollapsed
                        // collapsed, the bar goes away and takes its gutter
                        // with it
                        scrollBarEnabled: !root.consolePanelCollapsed

                        ListView {
                            spacing: 0

                            model: root.workspace ? root.workspace.consoleHistoryEntries : []
                            delegate: consoleRow
                        }
                    }

                    Connections {
                        target: workspace
                        function onConsoleHistoryEntriesChanged() {
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
                            root.consolePanelWidth = root.width;
                        }
                    }
                }
            }

            Timer {
                interval: 100
                running: true
                repeat: true
                onTriggered: {
                    root.workspace.syncConsole();
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
