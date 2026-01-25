import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

MenuBar {
    id: root

    height: Math.ceil(Scaling.pointSize * 2 * Scaling.uiScale)
    implicitHeight: height

    background: Rectangle {
        color: DarkPalette.base
    }

    delegate: MenuBarItem {
        font: Scaling.uiFont
        height: root.height
        padding: Math.round(6 * Scaling.uiScale)
    }

    Menu {
        title: qsTr("&File")
        popupType: Popup.Item

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
        popupType: Popup.Item

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
            shortcut: StandardKey.Preferences
            onTriggered: settingsWindow.open()
        }
    }

    Menu {
        title: qsTr("&Window")
        popupType: Popup.Item

        delegate: MenuItem {
            font: Scaling.uiFont
            padding: Math.round(6 * Scaling.uiScale)
        }

        Action {
            id: devicesMenuToggle
            text: qsTr("&Devices")
            checkable: true
            checked: devicesWindow !== null && devicesWindow.isOpen
            onToggled: {
                if (!devicesWindow)
                    return;
                if (checked)
                    devicesWindow.open();
                else
                    devicesWindow.forceClose(); // or close()
            }
        }
    }

    Menu {
        title: qsTr("&Help")
        popupType: Popup.Item

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
