import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

KDDW.DockWidget {
    id: root
    uniqueName: "alignmentWindow"
    title: "Alignment"

    property var workspace: null

    enum WindowMode {
        Snapshot,
        Picking,
        Refinement
    }

    enum ActiveView {
        Primary,
        Secondary
    }

    QtObject {
        id: windowState

        property int primaryDeviceIndex: 0
        property int secondaryDeviceIndex: 1

        property int mode: AlignmentWindow.WindowMode.Snapshot
        property int activeView: AlignmentWindow.ActiveView.Primary

        // completed pairs: [ { primary: vector3d, secondary: vector3d }, ... ]
        property var pairs: []

        // partial pick waiting for its partner
        property var pendingPrimaryPick: null
    }

    // extract per-view marker positions from pair data
    function primaryMarkerPositions() {
        var positions = [];
        for (var i = 0; i < windowState.pairs.length; ++i)
            positions.push(windowState.pairs[i].primary);
        if (windowState.pendingPrimaryPick !== null)
            positions.push(windowState.pendingPrimaryPick);
        return positions;
    }

    function secondaryMarkerPositions() {
        var positions = [];
        for (var i = 0; i < windowState.pairs.length; ++i)
            positions.push(windowState.pairs[i].secondary);
        return positions;
    }

    function handlePrimaryPicked(position) {
        windowState.pendingPrimaryPick = position;
        windowState.activeView = AlignmentWindow.ActiveView.Secondary;
        updateMarkers();
    }

    function handleSecondaryPicked(position) {
        var newPairs = windowState.pairs.slice();
        newPairs.push({
            primary: windowState.pendingPrimaryPick,
            secondary: position
        });
        windowState.pairs = newPairs;
        windowState.pendingPrimaryPick = null;
        windowState.activeView = AlignmentWindow.ActiveView.Primary;
        updateMarkers();
    }

    function undoLastPick() {
        if (windowState.activeView === AlignmentWindow.ActiveView.Secondary && windowState.pendingPrimaryPick !== null) {
            windowState.pendingPrimaryPick = null;
            windowState.activeView = AlignmentWindow.ActiveView.Primary;
        } else if (windowState.pairs.length > 0) {
            var newPairs = windowState.pairs.slice();
            var removed = newPairs.pop();
            windowState.pairs = newPairs;
            windowState.pendingPrimaryPick = removed.primary;
            windowState.activeView = AlignmentWindow.ActiveView.Secondary;
        }
        updateMarkers();
    }

    function updateMarkers() {
        var primary = windowState.pairs.map(p => p.primary);
        if (windowState.pendingPrimaryPick !== null)
            primary.push(windowState.pendingPrimaryPick);
        primaryDeviceView.markerPositions = primary;
        primaryDeviceView.pairCount = windowState.pairs.length;

        secondaryDeviceView.markerPositions = windowState.pairs.map(p => p.secondary);
        secondaryDeviceView.pairCount = windowState.pairs.length;
    }

    function resetAlignment() {
        windowState.mode = AlignmentWindow.WindowMode.Snapshot;
        windowState.activeView = AlignmentWindow.ActiveView.Primary;
        windowState.pairs = [];
        windowState.pendingPrimaryPick = null;
        primaryDeviceView.reset();
        secondaryDeviceView.reset();
    }

    Rectangle {
        anchors.fill: parent
        color: ThemeColors.base

        StackLayout {
            id: viewportStack
            currentIndex: windowState.mode < AlignmentWindow.WindowMode.Refinement ? 0 : 1
            anchors {
                left: parent.left
                right: parent.right
                top: parent.top
                bottom: controlBarStack.top
            }

            RowLayout {
                AlignmentView {
                    id: primaryDeviceView
                    visible: windowState.mode < AlignmentWindow.WindowMode.Refinement
                    deviceAdapter: root.workspace && root.workspace.deviceAdapters.length > windowState.primaryDeviceIndex ? root.workspace.deviceAdapters[windowState.primaryDeviceIndex] : null
                    live: windowState.mode == AlignmentWindow.WindowMode.Snapshot
                    enablePicking: windowState.mode == AlignmentWindow.WindowMode.Picking
                    active: enablePicking && windowState.activeView === AlignmentWindow.ActiveView.Primary
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    onPicked: position => root.handlePrimaryPicked(position)
                }

                Rectangle {
                    width: 1
                    Layout.fillHeight: true
                    color: ThemeColors.middark
                }

                AlignmentView {
                    id: secondaryDeviceView
                    visible: windowState.mode < AlignmentWindow.WindowMode.Refinement
                    deviceAdapter: root.workspace && root.workspace.deviceAdapters.length > windowState.secondaryDeviceIndex ? root.workspace.deviceAdapters[windowState.secondaryDeviceIndex] : null
                    live: windowState.mode == AlignmentWindow.WindowMode.Snapshot
                    enablePicking: windowState.mode == AlignmentWindow.WindowMode.Picking
                    active: enablePicking && windowState.activeView === AlignmentWindow.ActiveView.Secondary
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    onPicked: position => root.handleSecondaryPicked(position)
                }
            }

            RefinementView {
                id: refinementView
                visible: windowState.mode == AlignmentWindow.WindowMode.Refinement
                primaryAdapter: root.workspace && root.workspace.deviceAdapters.length > windowState.primaryDeviceIndex ? root.workspace.deviceAdapters[windowState.primaryDeviceIndex] : null
                secondaryAdapter: root.workspace && root.workspace.deviceAdapters.length > windowState.secondaryDeviceIndex ? root.workspace.deviceAdapters[windowState.secondaryDeviceIndex] : null
            }
        }

        StackLayout {
            id: controlBarStack
            currentIndex: windowState.mode
            height: Math.round(Scaling.uiScale * 52)
            anchors {
                left: parent.left
                right: parent.right
                bottom: parent.bottom
                bottomMargin: Math.round(Scaling.uiScale * 4)
            }

            RowLayout {
                id: shapshotControls
                spacing: Math.round(Scaling.uiScale * 16)

                Item {
                    Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
                }

                Component {
                    id: deviceSelectorItem
                    Text {
                        text: modelData.value("id") || modelData

                        font: Scaling.uiFont
                        color: ThemeColors.text

                        topPadding: 6
                        leftPadding: 6
                        rightPadding: 6
                        bottomPadding: 6

                        verticalAlignment: Text.AlignVCenter
                        elide: Text.ElideRight
                    }
                }

                Column {
                    spacing: Math.round(Scaling.uiScale * 4)
                    Label {
                        text: "Primary Device"
                    }

                    ComboBox {
                        id: primaryDeviceList
                        flat: true
                        model: root.workspace ? root.workspace.deviceAdapters : []
                        currentIndex: windowState.primaryDeviceIndex
                        textRole: "id"

                        contentItem: Text {
                            text: primaryDeviceList.displayText
                            font: Scaling.uiFont
                            color: ThemeColors.text
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                        }

                        delegate: ItemDelegate {
                            width: primaryDeviceList.width
                            text: modelData.id
                            font: Scaling.uiFont
                            highlighted: primaryDeviceList.highlightedIndex === index
                            enabled: secondaryDeviceList.currentIndex !== index
                        }

                        onActivated: windowState.primaryDeviceIndex = currentIndex
                    }
                }

                Column {
                    spacing: Math.round(Scaling.uiScale * 4)
                    Label {
                        text: "Secondary Device"
                    }
                    ComboBox {
                        id: secondaryDeviceList
                        flat: true
                        model: root.workspace ? root.workspace.deviceAdapters : []
                        currentIndex: windowState.secondaryDeviceIndex
                        textRole: "id"

                        contentItem: Text {
                            text: secondaryDeviceList.displayText
                            font: Scaling.uiFont
                            color: ThemeColors.text
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                        }

                        delegate: ItemDelegate {
                            width: secondaryDeviceList.width
                            text: modelData.id
                            font: Scaling.uiFont
                            highlighted: secondaryDeviceList.highlightedIndex === index
                            enabled: primaryDeviceList.currentIndex !== index
                        }

                        onActivated: windowState.secondaryDeviceIndex = currentIndex
                    }
                }

                Item {
                    Layout.fillWidth: true
                }

                IconButton {
                    id: snapshotButton
                    text: "Snapshot"
                    tooltip: "Capture a frame from the target devices and begin alignment"
                    onClicked: {
                        windowState.mode = AlignmentWindow.WindowMode.Picking;
                        windowState.activeView = AlignmentWindow.ActiveView.Primary;
                        primaryDeviceView.snapshot();
                        secondaryDeviceView.snapshot();
                    }
                }

                Item {
                    Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
                }
            }

            RowLayout {
                id: pickingControls
                spacing: Math.round(Scaling.uiScale * 12)

                Item {
                    Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
                }

                IconButton {
                    id: resetButton
                    text: "Reset"
                    tooltip: "Remove current alignment data and start over"
                    onClicked: root.resetAlignment()
                }

                IconButton {
                    text: "Undo last pick"
                    tooltip: "Undo the most recent point pick"
                    iconSource: FontAwesome.icon("solid/rotate-left")
                    enabled: windowState.pendingPrimaryPick !== null || windowState.pairs.length > 0
                    onClicked: root.undoLastPick()
                }

                Label {
                    text: windowState.pairs.length + " pair" + (windowState.pairs.length !== 1 ? "s" : "")
                    color: ThemeColors.text
                }

                Item {
                    Layout.fillWidth: true
                }

                IconButton {
                    id: alignButton
                    text: "Align"
                    tooltip: "Finish picking pairs and compute coarse transformation"
                    enabled: windowState.pairs.length >= 3
                    onClicked: {
                        windowState.mode = AlignmentWindow.WindowMode.Refinement;
                        refinementView.activate();
                    }
                }

                Item {
                    Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
                }
            }

            RowLayout {
                id: refinementControls
                spacing: Math.round(Scaling.uiScale * 12)

                Item {
                    Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
                }

                IconButton {
                    id: returnToPickingButton
                    text: "Picking"
                    tooltip: "Return to keypoint pair picking"
                    iconSource: FontAwesome.icon("solid/arrow-left-long")
                    onClicked: {
                        windowState.mode = AlignmentWindow.WindowMode.Picking;
                    }
                }

                Item {
                    Layout.fillWidth: true
                }

                IconButton {
                    id: applyButton
                    text: "Apply to Session"
                    tooltip: "Apply alignment transform to session configuration"
                    iconSource: FontAwesome.icon("solid/floppy-disk")
                    onClicked: {}
                }

                Item {
                    Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
                }
            }
        }
    }
}
