import QtQuick
import QtQuick.Controls

MenuBar {
    id: root

    Menu {
        title: qsTr("&File")
        popupType: Popup.Window

        Action {
            text: qsTr("&New Workspace")
            shortcut: StandardKey.New
            // onTriggered: workspaceModel.newWorkspace()
        }

        Action {
            text: qsTr("New &Session")
            shortcut: StandardKey.AddTab
            // onTriggered: workspaceModel.newWorkspace()
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
            text: qsTr("Save &As...")
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
        popupType: Popup.Window

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
    }

    Menu {
        title: qsTr("&Window")
        popupType: Popup.Native

        Action {
            id: devicesMenuToggle
            text: qsTr("&Devices")
            checkable: true
            onTriggered: {
                if (devicesWindow.isOpen)
                    devicesWindow.forceClose();
                else
                    devicesWindow.open();
            }
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
        title: qsTr("&Help")
        popupType: Popup.Native

        Action {
            text: qsTr("&About")
            shortcut: StandardKey.HelpContents
            onTriggered: aboutPopup.open()
        }
    }
}
