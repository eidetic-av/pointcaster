import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

MenuBar {
    id: root

    height: Math.ceil(Scaling.pointSize * Scaling.uiScale + 16 * Scaling.uiScale)
    implicitHeight: height

    background: Rectangle {
        color: ThemeColors.base
    }

    delegate: MenuBarItem {
        font: Scaling.uiFont
        height: root.height
        padding: Math.round(6 * Scaling.uiScale)
    }

    Menu {
        title: qsTr("&File")
        popupType: Popup.Native

        delegate: MenuItem {
            font: Scaling.uiFont
            padding: Math.round(6 * Scaling.uiScale)
        }

        Action {
            text: qsTr("&New Workspace")
            shortcut: StandardKey.New
        }

        Action {
            text: qsTr("New &Session")
            shortcut: StandardKey.AddTab
        }

        MenuSeparator {}

        Action {
            text: qsTr("&Open")
            shortcut: StandardKey.Open
            onTriggered: openWorkspaceDialog.open()
        }

        MenuSeparator {}

        Action {
            text: qsTr("&Save")
            shortcut: StandardKey.Save
            onTriggered: workspaceModel.save(true)
        }

        Action {
            text: qsTr("Save &As…")
            shortcut: StandardKey.SaveAs
            onTriggered: saveAsWorkspaceDialog.open()
        }

        MenuSeparator {}

        Action {
            text: qsTr("&Quit")
            shortcut: StandardKey.Quit
            onTriggered: workspaceModel.close()
        }
    }

    Menu {
        title: qsTr("&Edit")
        popupType: Popup.Native

        delegate: MenuItem {
            id: item
            font: Scaling.uiFont
            padding: Math.round(6 * Scaling.uiScale)
            opacity: enabled ? 1.0 : 0.4
        }

        Action {
            id: undoAction
            text: qsTr("&Undo")
            shortcut: StandardKey.Undo
            enabled: workspaceModel && workspaceModel.undoStack && workspaceModel.undoStack.canUndo
            onTriggered: workspaceModel.undoStack.undo()
        }

        Action {
            id: redoAction
            text: qsTr("&Redo")
            shortcut: StandardKey.Redo
            enabled: workspaceModel && workspaceModel.undoStack && workspaceModel.undoStack.canRedo
            onTriggered: workspaceModel.undoStack.redo()
        }

        MenuSeparator {}

        Action {
            id: settingsWindowEntry
            text: qsTr("&Preferences")
            shortcut: "Ctrl+,"
            onTriggered: settingsWindow.open()
        }
    }

    Menu {
        title: qsTr("&Window")
        popupType: Popup.Native

        delegate: MenuItem {
            font: Scaling.uiFont
            padding: Math.round(6 * Scaling.uiScale)
        }

        Action {
            id: devicesMenuToggle
            text: qsTr("&Devices")
            checkable: true
            checked: false

            onTriggered: {
                if (!devicesWindow)
                    return;

                if (devicesWindow.isOpen)
                    devicesWindow.forceClose();
                else
                    devicesWindow.open();
            }
        }

        Connections {
            target: devicesWindow
            function onIsOpenChanged() {
                devicesMenuToggle.checked = devicesWindow.isOpen;
            }
        }

        onAboutToShow: {
            if (devicesWindow)
                devicesMenuToggle.checked = devicesWindow.isOpen;
        }

        Component.onCompleted: {
            if (devicesWindow)
                devicesMenuToggle.checked = devicesWindow.isOpen;
        }
    }

    Menu {
        title: qsTr("&Help")
        popupType: Popup.Native

        delegate: MenuItem {
            font: Scaling.uiFont
            padding: Math.round(6 * Scaling.uiScale)
        }

        Action {
            text: qsTr("&About")
            shortcut: StandardKey.HelpContents
            onTriggered: aboutPopup.open()
        }
    }
}
