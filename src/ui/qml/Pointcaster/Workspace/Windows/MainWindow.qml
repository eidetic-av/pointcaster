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

    palette: DarkPalette.palette

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
        nameFilters: [qsTr("JSON files (*.json)"), qsTr("All files (*)")]
        onAccepted: workspaceModel.loadFromFile(selectedFile)
    }

    FileDialog {
        id: saveAsWorkspaceDialog
        fileMode: FileDialog.SaveFile
        nameFilters: [qsTr("JSON files (*.json)"), qsTr("All files (*)")]
        onAccepted: {
            workspaceModel.saveFileUrl = selectedFile;
            workspaceModel.save();
        }
    }

    Connections {
        target: workspaceModel
        function openSaveAsDialog() {
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

        KDDW.DockWidget {
            id: session2
            uniqueName: "session2"
            title: "session_2"
            affinities: ["view"]
            SessionView {}
        }

        Component.onCompleted: {
            // layoutSaver.restoreFromFile('mylayout.json');

            addDockWidgetAsTab(session1);
            addDockWidgetAsTab(session2);

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
        // popupType: Popup.Window
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

    SettingsWindow {
        id: settingsWindow
        x: root.x + (root.width - width) / 2
        y: root.y + (root.height - height) / 2
        // workspace: workspaceModel
    }
}
