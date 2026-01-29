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
        nameFilters: [qsTr("TOML files (*.toml)"), qsTr("All files (*)")]
        onAccepted: workspaceModel.loadFromFile(selectedFile)
    }

    FileDialog {
        id: saveAsWorkspaceDialog
        fileMode: FileDialog.SaveFile
        nameFilters: [qsTr("TOML files (*.toml)"), qsTr("All files (*)")]
        onAccepted: {
            const hasSavePath = workspaceModel && workspaceModel.saveFileUrl.toString() !== "";
            workspaceModel.saveFileUrl = selectedFile;
            // if there was no existing save path, we allow the 'save as' dialog to set the
            // current save path because it hasnt been set yet
            workspaceModel.save(!hasSavePath);
        }
    }

    Connections {
        target: workspaceModel
        function onOpenSaveAsDialog() {
            console.log("should have raised");
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

        // persistentCentralItemFileName: ":/qt/qml/Pointcaster/Workspace/Windows/Sessions/CentralSessionView.qml"

        options: KDDW.KDDockWidgets.MainWindowOption_HasCentralGroup
        // centralGroup: CentralSessionView {}

        DevicesWindow {
            id: devicesWindow
            workspace: workspaceModel
            affinities: ["edit"]
        }

        KDDW.DockWidget {
            id: session1
            uniqueName: "session1"
            title: "session_1"
            affinities: ["view"]
            SessionView {}
        }

        Component.onCompleted: {
            // layoutSaver.restoreFromFile('mylayout.json');

            addDockWidgetAsTab(session1);

            addDockWidget(devicesWindow, KDDW.KDDockWidgets.Location_OnLeft, null, Qt.size(400, 400));

            // addDockWidget(session2, KDDW.KDDockWidgets.Location_OnLeft, null, Qt.size(800,800), KDDW.DockWidget.NoFloat);
            // addDockWidget(session1, KDDW.KDDockWidgets.Location_OnRight);
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
        // workspace: workspaceModel
    }
}
