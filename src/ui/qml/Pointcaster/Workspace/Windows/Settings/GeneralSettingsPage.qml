import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    implicitWidth: 520 * Scaling.uiScale
    implicitHeight: 360 * Scaling.uiScale

    ScrollView {
        id: scroll
        anchors.fill: parent
        clip: true

        Pane {
            width: scroll.availableWidth
            padding: 16 * Scaling.uiScale

            ColumnLayout {
                width: parent.width
                spacing: 14 * Scaling.uiScale

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4 * Scaling.uiScale

                    Label {
                        text: "General"
                        font: Scaling.uiHeaderFont
                    }

                    Label {
                        text: "Application-wide preferences"
                        font: Scaling.uiFont
                        opacity: 0.7
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }

                GroupBox {
                    title: "Application"
                    Layout.fillWidth: true
                    font: Scaling.uiFont

                    ColumnLayout {
                        spacing: 10 * Scaling.uiScale

                        CheckBox {
                            text: "Restore previous session on startup"
                            font: Scaling.uiFont
                            checked: AppSettings.restoreLastSession
                            onToggled: AppSettings.restoreLastSession = checked
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10 * Scaling.uiScale

                            Label {
                                text: "Log level"
                                font: Scaling.uiFont
                                Layout.preferredWidth: 120 * Scaling.uiScale
                                opacity: 0.9
                            }

                            ComboBox {
                                id: logLevelCombo
                                Layout.fillWidth: true
                                font: Scaling.uiFont
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

                                Component.onCompleted: currentIndex = indexOfValue(AppSettings.logLevel)

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

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10 * Scaling.uiScale

                            Label {
                                text: "UI scale"
                                font: Scaling.uiFont
                                Layout.preferredWidth: 120 * Scaling.uiScale
                                opacity: 0.9
                            }

                            Slider {
                                id: uiScaleSlider
                                Layout.fillWidth: true

                                from: 0.5
                                to: 2.5

                                // Not applied in realtime: slider is "pending" only.
                                value: pendingUiScale

                                // Base step shown on the control; we implement the Ctrl=0.01 behaviour ourselves.
                                stepSize: 0.05

                                onMoved: {
                                    const fine = (Qt.keyboardModifiers & Qt.ControlModifier) !== 0;
                                    const step = fine ? 0.01 : 0.05;
                                    pendingUiScale = quantise(value, step);
                                }

                                Keys.onPressed: event => {
                                    const fine = (event.modifiers & Qt.ControlModifier) !== 0;
                                    const step = fine ? 0.01 : 0.05;

                                    if (event.key === Qt.Key_Left || event.key === Qt.Key_Down) {
                                        pendingUiScale = clamp(pendingUiScale - step);
                                        event.accepted = true;
                                    } else if (event.key === Qt.Key_Right || event.key === Qt.Key_Up) {
                                        pendingUiScale = clamp(pendingUiScale + step);
                                        event.accepted = true;
                                    } else if (event.key === Qt.Key_Home) {
                                        pendingUiScale = from;
                                        event.accepted = true;
                                    } else if (event.key === Qt.Key_End) {
                                        pendingUiScale = to;
                                        event.accepted = true;
                                    }
                                }
                            }

                            Label {
                                text: pendingUiScale.toFixed(2) + "×"
                                font: Scaling.uiFont
                                Layout.preferredWidth: 60 * Scaling.uiScale
                                horizontalAlignment: Text.AlignRight
                            }

                            Button {
                                id: applyScaleButton
                                text: "Apply"
                                font: Scaling.uiFont

                                enabled: Math.abs(pendingUiScale - AppSettings.uiScale) > 1e-9

                                onClicked: AppSettings.uiScale = pendingUiScale
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

    // --- local pending state / helpers ---

    property real pendingUiScale: AppSettings.uiScale

    function clamp(v) {
        return Math.max(uiScaleSlider.from, Math.min(uiScaleSlider.to, v));
    }

    function quantise(v, step) {
        const c = clamp(v);
        const q = Math.round(c / step) * step;
        // Avoid -0.00 and keep stability
        return Number(clamp(q).toFixed(2));
    }

    Connections {
        target: AppSettings
        function onUiScaleChanged() {
            // If something else changes the setting, keep the pending in sync.
            pendingUiScale = AppSettings.uiScale;
        }
    }
}
