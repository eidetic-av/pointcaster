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

                    Label {
                        text: "General"
                        font.pixelSize: 18
                    }

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
                            checked: AppSettings.restoreLastSession
                            onToggled: AppSettings.restoreLastSession = checked
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Label {
                                text: "Log level"
                                Layout.preferredWidth: 120
                                opacity: 0.9
                            }

                            ComboBox {
                                id: logLevelCombo
                                Layout.fillWidth: true
                                textRole: "text"
                                valueRole: "value"

                                model: [
                                    {
                                        text: "Trace",
                                        value: AppSettings.Trace
                                    },
                                    {
                                        text: "Debug",
                                        value: AppSettings.Debug
                                    },
                                    {
                                        text: "Info",
                                        value: AppSettings.Info
                                    },
                                    {
                                        text: "Warn",
                                        value: AppSettings.Warn
                                    },
                                    {
                                        text: "Error",
                                        value: AppSettings.Error
                                    },
                                    {
                                        text: "Critical",
                                        value: AppSettings.Critical
                                    },
                                    {
                                        text: "Off",
                                        value: AppSettings.Off
                                    }
                                ]

                                Component.onCompleted: {
                                    currentIndex = indexOfValue(AppSettings.logLevel);
                                }

                                onActivated: {
                                    const v = currentValue;
                                    if (v !== undefined && v !== null)
                                        AppSettings.logLevel = v;
                                }

                                Connections {
                                    target: AppSettings
                                    function onLogLevelChanged() {
                                        const idx = logLevelCombo.indexOfValue(AppSettings.logLevel);
                                        if (idx >= 0 && idx !== logLevelCombo.currentIndex)
                                            logLevelCombo.currentIndex = idx;
                                    }
                                }
                            }
                        }
                    }
                }

                Item {
                    Layout.fillHeight: true
                }
            }
        }
    }
}
