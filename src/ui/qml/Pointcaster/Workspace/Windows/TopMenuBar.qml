import QtQuick
import QtQuick.Controls

MenuBar {

        Menu {
            title: qsTr("&File")
            popupType: Popup.Window

            Action {
                text: qsTr("&New Workspace")
            }
            Action {
                text: qsTr("New &Session")
            }

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
                onTriggered: settingsWindow.open();
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