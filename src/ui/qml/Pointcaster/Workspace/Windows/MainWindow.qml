import QtQuick
import QtQuick.Controls.Fusion
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

ApplicationWindow {
    id: root
    visible: true
    title: "Pointcaster"

    minimumWidth: 1200
    minimumHeight: 800

    palette: ThemeColors.palette

    FontLoader {
        source: "../Resources/Fonts/AtkinsonHyperlegibleNext-Light.otf"
    }
    FontLoader {
        source: "../Resources/Fonts/AtkinsonHyperlegibleNext-Regular.otf"
    }
    FontLoader {
        source: "../Resources/Fonts/AtkinsonHyperlegibleNext-Medium.otf"
    }
    FontLoader {
        source: "../Resources/Fonts/AtkinsonHyperlegibleNext-Bold.otf"
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
    }

    function toggleWindow(target) {
        if (target.dockWidget.isOpen) {
            target.dockWidget.forceClose();
        } else {
            target.dockWidget.show();
        }
    }

    menuBar: TopMenuBar {}

    KDDW.DockingArea {
        id: dockingArea
        anchors.fill: parent
        uniqueName: "MainDockingArea"

        affinities: ["edit", "view"]
        options: KDDW.KDDockWidgets.MainWindowOption_HasCentralGroup

        DevicesWindow {
            id: devicesWindow
            workspace: workspaceModel
            affinities: ["edit"]
        }

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
                    // bind directly to the DockWidget property; don't rely on parent lookup
                    sessionAdapter: sessionDockWidget.sessionAdapter
                }
            }
        }

        // id(string) -> KDDW.DockWidget
        property var sessionDockById: ({})

        function syncSessionWindows() {
            const adapters = workspaceModel.sessionAdapters;
            console.log(`syncSessionWindows: adapters length=${adapters.length}`);

            // Mark all existing as unseen initially
            const seen = ({});
            for (const key in sessionDockById)
                seen[key] = false;

            for (let i = 0; i < adapters.length; ++i) {
                const adapter = adapters[i];
                if (!adapter)
                    continue;

                const idStr = String(adapter.id);
                if (idStr.length === 0)
                    continue;

                seen[idStr] = true;

                let dock = sessionDockById[idStr];
                if (dock) {
                    // Update existing dock
                    const newTitle = String(adapter.label);
                    if (dock.title !== newTitle)
                        dock.title = newTitle;

                    dock.sessionAdapter = adapter;

                    continue;
                }

                // Create new dock
                const newDock = sessionWindowComponent.createObject(dockingArea, {
                    dockUniqueName: idStr,
                    dockTitle: String(adapter.label),
                    sessionAdapter: adapter
                });

                if (!newDock) {
                    console.log(`Failed to create dock for session id=${idStr}`);
                    continue;
                }

                sessionDockById[idStr] = newDock;

                // Place it (tab with central group)
                addDockWidgetAsTab(newDock);
            }

            // Remove docks whose sessions no longer exist
            for (const id in sessionDockById) {
                if (seen[id])
                    continue;

                const dock = sessionDockById[id];
                console.log(`Removing dock for deleted session id=${id}`);

                // Close/remove from layout first
                try {
                    if (dock && dock.isOpen)
                        dock.forceClose();
                } catch (e) {}

                if (dock)
                    dock.destroy();

                delete sessionDockById[id];
            }
        }

        Component.onCompleted: {
            addDockWidget(devicesWindow, KDDW.KDDockWidgets.Location_OnLeft, null, Qt.size(400, 400));
            dockingArea.syncSessionWindows();
        }

        Connections {
            target: workspaceModel
            function onSessionAdaptersChanged() {
                dockingArea.syncSessionWindows();
            }
        }
    }

    KDDW.LayoutSaver {
        id: layoutSaver
    }

    Connections {
        target: Qt.application
        function onAboutToQuit() {
            console.log("App is quitting");
            // layoutSaver.saveToFile('mylayout.json');
        }
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
        x: root.x + (root.width - width) / 2
        y: root.y + (root.height - height) / 2
    }
}
