import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Dialog {
    id: root
    title: "Preferences"

    popupType: Popup.Window
    modal: true
    focus: true
    dim: true
    visible: false

    implicitWidth: 720
    implicitHeight: 480

    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
    standardButtons: Dialog.Close

    contentItem: RowLayout {
        spacing: 0

        ListView {
            id: sidebar
            Layout.preferredWidth: 180
            Layout.fillHeight: true
            clip: true

            model: SettingsPageRegistry
            currentIndex: Math.max(0, SettingsPageRegistry.indexOfKey(SettingsPageRegistry.defaultKey))

            delegate: ItemDelegate {
                width: ListView.view.width
                text: model.title
                highlighted: ListView.isCurrentItem

                onClicked: {
                    root.setPageIndex(index);
                }
            }
        }

        Rectangle {
            width: 1
            Layout.fillHeight: true
            color: Qt.darker(palette.window, 1.15)
        }

        StackView {
            id: stack
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            // transitions between pages happen by default,
            // just disable them for now
            replaceEnter: Transition { }
            replaceExit: Transition { }
        }
    }

    function setPageIndex(newIndex) {
        if (newIndex < 0 || newIndex >= SettingsPageRegistry.count)
            return;
        sidebar.currentIndex = newIndex;
        const url = SettingsPageRegistry.pageUrlAt(newIndex);
        if (url && url.toString().length > 0)
            stack.replace(url);
    }

    function setActivePage(key) {
        const idx = SettingsPageRegistry.indexOfKey(key);
        if (idx < 0)
            return;
        setPageIndex(idx);
    }

    function openPage(key) {
        setActivePage(key);
        open();
    }

    Component.onCompleted: {
        // core pages are added initially, other pages loaded via plugins are added using the same api
        SettingsPageRegistry.addPage("general", "General", "qrc:/qt/qml/Pointcaster/Workspace/Windows/Settings/GeneralSettingsPage.qml");
        SettingsPageRegistry.addPage("user_interface", "User Interface", "qrc:/qt/qml/Pointcaster/Workspace/Windows/Settings/UserInterfaceSettingsPage.qml");

        // initialilly set the stack view to show the first page
        sidebar.currentIndex = 0;
        const initialUrl = SettingsPageRegistry.pageUrlAt(0);
        stack.replace(initialUrl);
    }
}
