import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

MenuBar {
    id: root

    height: Math.ceil(Scaling.pointSize * Scaling.uiScale + 14 * Scaling.uiScale)
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
        popupType: Popup.Item

        delegate: MenuItem {
            font: Scaling.uiFont
            padding: Math.round(6 * Scaling.uiScale)
        }

        Action {
            text: qsTr("&New Workspace")
            shortcut: StandardKey.New
            onTriggered: workspaceModel.newWorkspace()
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
            shortcut: "Ctrl+,"
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
            id: devicesWindowMenuToggle
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
                devicesWindowMenuToggle.checked = devicesWindow.isOpen;
            }
        }

        Action {
            id: recordingWindowMenuToggle
            text: qsTr("&Recording")
            checkable: true
            checked: false

            onTriggered: {
                if (!recordingWindow)
                    return;

                if (recordingWindow.isOpen)
                    recordingWindow.forceClose();
                else
                    recordingWindow.open();
            }
        }

        Connections {
            target: recordingWindow
            function onIsOpenChanged() {
                recordingWindowMenuToggle.checked = recordingWindow.isOpen;
            }
        }


        property bool openedAlignWindow: false

        Action {
            id: alignmentWindowMenuToggle
            text: qsTr("&Alignment")
            checkable: true
            checked: false

            onTriggered: {
                if (!alignmentWindow)
                    return;

                if (alignmentWindow.isOpen)
                    alignmentWindow.forceClose();
                else {
                    alignmentWindow.open();
                    if (!openedAlignWindow) {
                        openedAlignWindow = true;
                        // TODO there must be a way to set this inside AlignmentWindow not here
                        alignmentWindow.resize(1640, 900);
                    }
                }
            }
        }

        Connections {
            target: alignmentWindow
            function onIsOpenChanged() {
                alignmentWindowMenuToggle.checked = alignmentWindow.isOpen;
            }
        }

        function sync() {
            if (devicesWindow)
                devicesWindowMenuToggle.checked = devicesWindow.isOpen;
            if (recordingWindow)
                recordingWindowMenuToggle.checked = recordingWindow.isOpen;
            if (alignmentWindow)
                alignmentWindowMenuToggle.checked = alignmentWindow.isOpen;
        }

        onAboutToShow: sync()
        Component.onCompleted: sync()
    }

    Menu {
        title: qsTr("&Help")
        popupType: Popup.Item

        delegate: MenuItem {
            font: Scaling.uiFont
            padding: Math.round(6 * Scaling.uiScale)
        }

        Action {
            text: qsTr("&Documentation")
            shortcut: StandardKey.HelpContents
            onTriggered: Qt.openUrlExternally("https://docs.pointcaster.net")
        }

        Action {
            text: qsTr("&About")
            onTriggered: aboutPopup.open()
        }
    }
}
