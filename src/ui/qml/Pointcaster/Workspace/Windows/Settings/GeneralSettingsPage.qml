import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    implicitWidth: 520
    implicitHeight: 360

    ScrollView {
        id: scroll
        anchors.fill: parent
        clip: true

        Pane {
            width: scroll.availableWidth
            padding: 16

            ColumnLayout {
                width: parent.width
                spacing: 14

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    Label { text: "General"; font.pixelSize: 18 }

                    Label {
                        text: "Application-wide preferences"
                        opacity: 0.7
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }

                GroupBox {
                    title: "Application"
                    Layout.fillWidth: true

                    ColumnLayout {
                        spacing: 10

                        CheckBox { 
                            text: "Restore previous session on startup" 
                            checked: AppSettings.restoreSession
                            onToggled: AppSettings.restoreSession = checked
                        }
                        CheckBox { text: "Check for updates automatically" }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Label { text: "Language"; Layout.preferredWidth: 120; opacity: 0.9 }

                            ComboBox {
                                Layout.fillWidth: true
                                model: ["System Default", "English", "Deutsch", "Français"]
                                currentIndex: 0
                            }
                        }
                    }
                }

                GroupBox {
                    title: "UI"
                    Layout.fillWidth: true

                    ColumnLayout {
                        spacing: 10

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Label { text: "Theme"; Layout.preferredWidth: 120; opacity: 0.9 }

                            ComboBox {
                                Layout.fillWidth: true
                                model: ["Dark", "Light", "System"]
                                currentIndex: 0
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Label { text: "UI scale"; Layout.preferredWidth: 120; opacity: 0.9 }

                            Slider {
                                id: uiScaleSlider
                                Layout.fillWidth: true
                                from: 0.75
                                to: 1.5
                                stepSize: 0.05
                                value: 1.0
                            }

                            Label {
                                text: Number(uiScaleSlider.value).toFixed(2) + "×"
                                horizontalAlignment: Text.AlignRight
                                Layout.preferredWidth: 56
                                opacity: 0.9
                            }

                            Button { text: "Reset"; onClicked: uiScaleSlider.value = 1.0 }
                        }

                        CheckBox { text: "Show tooltips"; checked: true }
                    }
                }

                GroupBox {
                    title: "Files"
                    Layout.fillWidth: true

                    ColumnLayout {
                        spacing: 10

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Label { text: "Default workspace location"; Layout.preferredWidth: 180; opacity: 0.9 }

                            TextField {
                                Layout.fillWidth: true
                                placeholderText: "/path/to/workspaces"
                            }

                            Button { text: "Browse…"; onClicked: {} }
                        }
                    }
                }

                Item { Layout.fillHeight: true }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Item { Layout.fillWidth: true }

                    Button {
                        text: "Reset all"
                        onClicked: {
                            // WorkspaceState.resetGeneralSettings()
                        }
                    }
                }
            }
        }
    }
}
