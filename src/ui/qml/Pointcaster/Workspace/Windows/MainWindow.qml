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

    FileDialog {
        id: saveAsWorkspaceDialog
        fileMode: FileDialog.SaveFile
        nameFilters: [
            qsTr("JSON files (*.json)"),
            qsTr("All files (*)")
        ]
        onAccepted: {
            workspaceModel.saveFileUrl = selectedFile
            workspaceModel.save()
        }
    }

    Connections {
        target: workspaceModel
        function openSaveAsDialog()
        {
            saveAsWorkspaceDialog.open()
        }
    }


    function toggleWindow(target)
    {
        if (target.dockWidget.isOpen)
        {
            target.dockWidget.forceClose();
        } else {
        target.dockWidget.show();
    }
}

menuBar : MenuBar {
    Menu {
        title: qsTr("&File")
        popupType: Popup.Native

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

        Action {
            text: qsTr("Save As...")
            onTriggered: saveAsWorkspaceDialog.open()
        }

        MenuSeparator {}

        Action {
            text: qsTr("&Quit")
            onTriggered: workspaceModel.close()
        }
    }

    Menu {
        title: qsTr("&Window")
        popupType: Popup.Native

        Action {
            id: devicesMenuToggle
            text: qsTr("&Devices")
            checkable: true
            onTriggered: {
                if (devicesWindow.isOpen) devicesWindow.forceClose()
                    else devicesWindow.open()
            }
        }

    }

    Menu {
        title: qsTr("&Help")
        popupType: Popup.Native

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

    SessionViewWindow {
        id: session0
    }

    DevicesWindow {
        id: devicesWindow
    }

    Component.onCompleted: {
        addDockWidget(session0, KDDW.KDDockWidgets.Location_OnLeft)
        addDockWidget(devicesWindow, KDDW.KDDockWidgets.Location_OnLeft)
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

}

