import QtQuick
import QtQuick.Controls.Fusion
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

ApplicationWindow {
    id: root
    visible: true
    title: "Pointcaster"

    minimumWidth: 1024
    minimumHeight: 600
    width: 1600
    height: 1066

    palette: ThemeColors.palette

    FontLoader {
        source: "../Fonts/AtkinsonHyperlegibleNext-Light.otf"
    }
    FontLoader {
        source: "../Fonts/AtkinsonHyperlegibleNext-Regular.otf"
    }
    FontLoader {
        source: "../Fonts/AtkinsonHyperlegibleNext-Medium.otf"
    }
    FontLoader {
        source: "../Fonts/AtkinsonHyperlegibleNext-Bold.otf"
    }

    FontLoader {
        source: "../Fonts/AtkinsonHyperlegibleMono-Light.otf"
    }
    FontLoader {
        source: "../Fonts/AtkinsonHyperlegibleMono-Regular.otf"
    }
    FontLoader {
        source: "../Fonts/AtkinsonHyperlegibleMono-Medium.otf"
    }
    FontLoader {
        source: "../Fonts/AtkinsonHyperlegibleMono-Bold.otf"
    }

    font.family: "Atkinson Hyperlegible Next"

    FileDialog {
        id: openWorkspaceDialog
        nameFilters: [qsTr("YAML files (*.yaml)"), qsTr("All files (*)")]
        onAccepted: workspaceModel.loadFromFile(selectedFile)
    }

    FileDialog {
        id: saveAsWorkspaceDialog
        fileMode: FileDialog.SaveFile
        nameFilters: [qsTr("YAML files (*.yaml)"), qsTr("All files (*)")]
        onAccepted: {
            const hasSavePath = workspaceModel && workspaceModel.saveFileUrl.toString() !== "";
            workspaceModel.saveFileUrl = selectedFile;
            workspaceModel.save(!hasSavePath);
        }
    }

    Connections {
        target: workspaceModel
        function onOpenSaveAsDialog() {
            saveAsWorkspaceDialog.open();
        }
        function onUiStateChanged() {
            Workspace.applyPersistentState(workspaceModel.uiState);
        }
    }

    Connections {
        target: Workspace
        function onLabelColumnWidthChanged() {
            workspaceModel.setUiStateValue("labelColumnWidth", Workspace.labelColumnWidth);
        }
        function onDeviceListHeightChanged() {
            workspaceModel.setUiStateValue("deviceListHeight", Workspace.deviceListHeight);
        }
        function onStreamChannelListHeightChanged() {
            workspaceModel.setUiStateValue("streamChannelListHeight", Workspace.streamChannelListHeight);
        }
    }

    function toggleWindow(target) {
        if (target.dockWidget.isOpen) {
            target.dockWidget.forceClose();
        } else {
            target.dockWidget.show();
        }
    }

    menuBar: TopMenuBar {}

    footer: StatusBar {}

    FocusReleaser {}

    KDDW.DockingArea {
        id: mainDockingArea
        anchors.fill: parent
        uniqueName: "MainDockingArea"

        affinities: ["edit", "view"]
        options: KDDW.KDDockWidgets.MainWindowOption_HasCentralGroup | KDDW.KDDockWidgets.MainWindowOption_CentralWidgetGetsAllExtraSpace

        Component {
            id: sessionWindowComponent

            KDDW.DockWidget {
                id: sessionDockWidget

                required property string dockUniqueName
                required property string dockTitle
                required property var sessionAdapter

                uniqueName: dockUniqueName
                title: dockTitle
                affinities: ["view"]

                SessionView {
                    id: sessionView
                    workspace: workspaceModel
                    sessionAdapter: sessionDockWidget.sessionAdapter
                    deviceAdapters: workspaceModel ? workspaceModel.deviceAdapters : []

                    Component.onCompleted: Workspace.addSessionView(this)
                    Component.onDestruction: Workspace.eraseSessionView(this)
                }

                onIsFocusedChanged: {
                    if (isFocused && sessionAdapter)
                        workspaceModel.selectedSessionId = String(sessionAdapter.id);
                }

                onIsOpenChanged: {
                    if (isOpen && isFocused && sessionAdapter)
                        workspaceModel.selectedSessionId = String(sessionAdapter.id);
                }
            }
        }

        DevicesWindow {
            id: devicesWindow
            workspace: workspaceModel
            affinities: ["edit"]

            onNodeSelected: {
                // update session view UIs when a new device is selected
                Workspace.sessionViews.forEach(sessionView => {
                    if (sessionView)
                        sessionView.selectionTransformUpdate();
                });
            }
        }

        SessionPropertiesWindow {
            id: sessionPropertiesWindow
            workspace: workspaceModel
            affinities: ["edit"]
        }

        RecordingWindow {
            id: recordingWindow
            workspace: workspaceModel
            recorder: workspaceModel.recorder
            affinities: ["edit"]
        }

        PublishersWindow {
            id: publishersWindow
            workspace: workspaceModel
            affinities: ["edit"]
        }

        StreamingWindow {
            id: streamingWindow
            workspace: workspaceModel
            affinities: ["edit"]
        }

        AlignmentWindow {
            id: alignmentWindow
            workspace: workspaceModel
        }

        // id(string) -> KDDW.DockWidget
        property var sessionDockById: ({})

        property bool initialSessionSyncDone: false

        function focusSession(sessionId) {
            const dock = sessionDockById[sessionId];
            if (!dock)
                return;
            if (!dock.isOpen)
                dock.open();
            dock.setAsCurrentTab();
            dock.raise();
            workspaceModel.selectedSessionId = sessionId;
        }

        function syncSessionWindows() {
            const sessionAdapters = workspaceModel.sessionAdapters;

            // Mark all existing as unseen initially
            const seen = ({});
            for (const key in sessionDockById)
                seen[key] = false;

            for (let i = 0; i < sessionAdapters.length; ++i) {
                const sessionAdapter = sessionAdapters[i];
                if (!sessionAdapter)
                    continue;

                const idStr = String(sessionAdapter.id);
                if (idStr.length === 0)
                    continue;

                seen[idStr] = true;

                let dock = sessionDockById[idStr];
                if (dock) {
                    // Update existing dock
                    const newTitle = String(sessionAdapter.label);
                    if (dock.title !== newTitle)
                        dock.title = newTitle;

                    dock.sessionAdapter = sessionAdapter;

                    continue;
                }

                // Create new dock
                const newDock = sessionWindowComponent.createObject(mainDockingArea, {
                    dockUniqueName: idStr,
                    dockTitle: String(sessionAdapter.label),
                    sessionAdapter: sessionAdapter
                });

                if (!newDock) {
                    console.log(`Failed to create dock for session id=${idStr}`);
                    continue;
                }

                sessionDockById[idStr] = newDock;

                // Place it (tab with central group)
                addDockWidgetAsTab(newDock);

                if (mainDockingArea.initialSessionSyncDone)
                    newDock.setAsCurrentTab();
            }

            // Remove docks whose sessions no longer exist
            for (const id in sessionDockById) {
                if (seen[id])
                    continue;

                const dock = sessionDockById[id];
                console.log(`Removing dock for deleted session id=${id}`);

                if (dock)
                    dock.deleteDockWidgetLater();

                delete sessionDockById[id];
            }
        }

        readonly property int sideColumnWidth: Math.round(350 * Scaling.uiScale)

        Component.onCompleted: {
            // initial layout

            const sideColumnSize = Qt.size(mainDockingArea.sideColumnWidth, 0);

            // -- right
            addDockWidget(sessionPropertiesWindow, KDDW.KDDockWidgets.Location_OnRight, null, sideColumnSize);
            sessionPropertiesWindow.addDockWidgetAsTab(publishersWindow);
            sessionPropertiesWindow.addDockWidgetAsTab(streamingWindow);
            sessionPropertiesWindow.setAsCurrentTab();

            // -- left
            addDockWidget(devicesWindow, KDDW.KDDockWidgets.Location_OnLeft, null, sideColumnSize);

            devicesWindow.addDockWidgetAsTab(recordingWindow);
            recordingWindow.close();

            devicesWindow.setAsCurrentTab();

            mainDockingArea.syncSessionWindows();
            mainDockingArea.initialSessionSyncDone = true;
        }

        Connections {
            target: workspaceModel
            function onSessionAdaptersChanged() {
                mainDockingArea.syncSessionWindows();
            }
            function onSessionAdded(sessionId) {
                mainDockingArea.focusSession(sessionId);
            }
        }
    }

    KDDW.LayoutSaver {
        id: layoutSaver
    }

    Popup {
        id: aboutPopup
        popupType: Popup.Native
        modal: true
        dim: true
        focus: true
        implicitWidth: 520 * Scaling.uiScale
        implicitHeight: 360 * Scaling.uiScale

        x: (parent.width - width) / 2
        y: (parent.height - height) / 2

        background: Rectangle {
            color: ThemeColors.alternateBase
            border.color: ThemeColors.middark
            radius: 3
        }

        Column {
            anchors.centerIn: parent
            spacing: 12 * Scaling.uiScale

            Label {
                text: "Pointcaster\n\nAbout dialog placeholder"
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                color: ThemeColors.text
            }

            Button {
                text: "Close"
                onClicked: aboutPopup.close()
            }
        }
    }

    SettingsWindow {
        id: settingsWindow
    }
}
