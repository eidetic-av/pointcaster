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

    font: Scaling.uiFont

    implicitWidth: Math.round(720 * Scaling.uiScale)
    implicitHeight: Math.round(480 * Scaling.uiScale)

    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    footer: Item {
        implicitHeight: footerLayout.implicitHeight
        width: parent.width

        RowLayout {
            id: footerLayout
            anchors.right: parent.right
            anchors.rightMargin: Math.round(12 * Scaling.uiScale)
            anchors.verticalCenter: parent.verticalCenter
            spacing: Math.round(8 * Scaling.uiScale)

            IconButton {
                text: qsTr("Close")
                onClicked: root.close()
            }
        }
    }

    contentItem: RowLayout {
        spacing: 0

        ListView {
            id: sidebar
            Layout.preferredWidth: Math.round(180 * Scaling.uiScale)
            Layout.fillHeight: true
            clip: true

            model: SettingsPageRegistry
            currentIndex: Math.max(0, SettingsPageRegistry.indexOfKey(SettingsPageRegistry.defaultKey))

            delegate: ItemDelegate {
                width: ListView.view.width
                text: model.title
                font: Scaling.uiFont
                highlighted: ListView.isCurrentItem
                padding: Math.round(8 * Scaling.uiScale)

                onClicked: root.setPageIndex(index)
            }
        }

        Rectangle {
            width: Math.max(1, Math.round(1 * Scaling.uiScale))
            Layout.fillHeight: true
            color: Qt.darker(palette.window, 1.15)
        }

        StackView {
            id: stack
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            replaceEnter: Transition {}
            replaceExit: Transition {}
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
        SettingsPageRegistry.addPage("general", "General", "qrc:/qt/qml/Pointcaster/Workspace/Windows/Settings/GeneralSettingsPage.qml");
        SettingsPageRegistry.addPage("network", "Network", "qrc:/qt/qml/Pointcaster/Workspace/Windows/Settings/NetworkSettingsPage.qml");

        sidebar.currentIndex = 0;
        const initialUrl = SettingsPageRegistry.pageUrlAt(0);
        stack.replace(initialUrl);
    }
}
