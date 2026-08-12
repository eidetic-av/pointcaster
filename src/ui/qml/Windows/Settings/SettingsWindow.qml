import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Dialog {
    id: root
    title: "Preferences"

    popupType: Popup.Window
    focus: true
    modal: false
    dim: false
    visible: false

    font: Scaling.uiFont

    implicitWidth: Math.round(720 * Scaling.uiScale)
    implicitHeight: Math.round(480 * Scaling.uiScale)

    padding: 0

    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    background: Rectangle {
        color: ThemeColors.window
    }

    footer: Rectangle {
        id: footerContent

        implicitHeight: footerLayout.implicitHeight + Math.round(16 * Scaling.uiScale)
        color: ThemeColors.dark

        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            height: 1
            color: ThemeColors.middark
        }

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

    contentItem: Item {
        id: windowContent

        FocusReleaser {}

        RowLayout {
            anchors.fill: parent
            spacing: 0

            Rectangle {
                id: sidebarPane

                Layout.preferredWidth: Math.round(172 * Scaling.uiScale)
                Layout.fillHeight: true
                color: ThemeColors.dark

                ListView {
                    id: sidebar

                    anchors.fill: parent
                    topMargin: Math.round(6 * Scaling.uiScale)
                    bottomMargin: Math.round(6 * Scaling.uiScale)
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    keyNavigationEnabled: false

                    model: SettingsPageRegistry

                    delegate: Item {
                        id: entry

                        required property int index
                        required property string title
                        required property bool isSection

                        readonly property bool current: ListView.isCurrentItem
                        readonly property real rowHeight: Math.max(Math.round(30 * Scaling.uiScale), Math.ceil(Scaling.pointSize * 2.1))

                        width: ListView.view.width

                        height: isSection ? rowHeight + Math.round(12 * Scaling.uiScale) : rowHeight

                        Label {
                            visible: entry.isSection

                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.bottom: parent.bottom
                            anchors.leftMargin: Math.round(11 * Scaling.uiScale)
                            anchors.rightMargin: Math.round(8 * Scaling.uiScale)
                            height: entry.rowHeight

                            text: entry.title
                            font: Scaling.uiFont
                            color: ThemeColors.withAlpha(ThemeColors.text, 0.5)
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                        }

                        ItemDelegate {
                            id: pageEntry

                            anchors.fill: parent
                            visible: !entry.isSection
                            padding: 0
                            leftPadding: Math.round(11 * Scaling.uiScale)
                            rightPadding: Math.round(8 * Scaling.uiScale)

                            onClicked: root.setPageIndex(entry.index)

                            background: Rectangle {
                                color: {
                                    if (pageEntry.hovered)
                                        return ThemeColors.midlight;
                                    if (entry.current)
                                        return ThemeColors.mid;
                                    return "transparent";
                                }
                            }

                            contentItem: Label {
                                text: entry.title
                                font: Scaling.uiFont
                                color: ThemeColors.text
                                verticalAlignment: Text.AlignVCenter
                                elide: Text.ElideRight
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.preferredWidth: Math.max(1, Math.round(1 * Scaling.uiScale))
                Layout.fillHeight: true
                color: ThemeColors.middark
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

    }

    function setPageIndex(newIndex) {
        const url = SettingsPageRegistry.pageUrlAt(newIndex);
        if (!url || url.toString().length === 0)
            return;

        sidebar.currentIndex = newIndex;
        stack.replace(url);
    }

    Component.onCompleted: {
        SettingsPageRegistry.addPage("general", "General", "qrc:/qt/qml/Pointcaster/Windows/Settings/GeneralSettingsPage.qml");
        SettingsPageRegistry.addPage("interface", "Interface", "qrc:/qt/qml/Pointcaster/Windows/Settings/InterfaceSettingsPage.qml");
        SettingsPageRegistry.addPage("performance", "Performance", "qrc:/qt/qml/Pointcaster/Windows/Settings/PerformanceSettingsPage.qml");
        SettingsPageRegistry.addPage("diagnostics", "Diagnostics", "qrc:/qt/qml/Pointcaster/Windows/Settings/DiagnosticsSettingsPage.qml");

        workspaceModel.registerPluginSettingsPages();

        setPageIndex(0);
    }
}
