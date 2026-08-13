import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

Window {
    id: root
    title: "Alignment"

    width: 1200
    height: 600

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

    FocusReleaser {}

    AlignmentController {
        id: alignmentController
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
                alignmentController: alignmentController
            }
        }

        StackLayout {
            id: controlBarStack
            currentIndex: windowState.mode
            height: windowState.mode === AlignmentWindow.WindowMode.Refinement ? Math.round(Scaling.uiScale * 80) : Math.round(Scaling.uiScale * 52)
            anchors {
                left: parent.left
                right: parent.right
                bottom: parent.bottom
                bottomMargin: Math.round(Scaling.uiScale * 4)
            }

            SnapshotControls {
                workspace: root.workspace
                primaryDeviceIndex: windowState.primaryDeviceIndex
                secondaryDeviceIndex: windowState.secondaryDeviceIndex

                onPrimaryDeviceChanged: index => windowState.primaryDeviceIndex = index
                onSecondaryDeviceChanged: index => windowState.secondaryDeviceIndex = index
                onSnapshotRequested: {
                    alignmentController.snapshotClouds(root.workspace.deviceAdapters[windowState.primaryDeviceIndex], root.workspace.deviceAdapters[windowState.secondaryDeviceIndex]);
                    windowState.mode = AlignmentWindow.WindowMode.Picking;
                    windowState.activeView = AlignmentWindow.ActiveView.Primary;
                    primaryDeviceView.snapshot();
                    secondaryDeviceView.snapshot();
                }
            }

            PickingControls {
                pairCount: windowState.pairs.length
                canUndo: windowState.pendingPrimaryPick !== null || windowState.pairs.length > 0

                onResetRequested: root.resetAlignment()
                onUndoRequested: root.undoLastPick()
                onAlignRequested: {
                    alignmentController.computeFromPairs(windowState.pairs);
                    if (alignmentController.hasResult) {
                        windowState.mode = AlignmentWindow.WindowMode.Refinement;
                        refinementView.activate();
                    }
                }
            }

            RefinementControls {
                alignmentController: alignmentController

                onBackToPicking: windowState.mode = AlignmentWindow.WindowMode.Picking
                onApplyRequested: {
                    if (!alignmentController.hasResult)
                        return;
                    var adapter = root.workspace.deviceAdapters[windowState.secondaryDeviceIndex];
                    // the result is a world-space delta...
                    // so transform to internal space
                    var local = root.workspace.localTransformForWorldAlignment(adapter.id, alignmentController.transform);
                    if (local.position === undefined)
                        return;
                    adapter.set("transform/position", local.position);
                    adapter.set("transform/rotation", local.rotation);
                }
            }
        }
    }
}
