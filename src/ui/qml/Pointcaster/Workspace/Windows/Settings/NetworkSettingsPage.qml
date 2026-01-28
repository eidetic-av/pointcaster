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

        // Never allow horizontal scrolling.
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        // Keep content width locked to the viewport width so children can't expand it.
        contentWidth: availableWidth

        Pane {
            // Use contentWidth (locked) rather than availableWidth (can differ in some styles)
            width: scroll.contentWidth
            padding: 16 * Scaling.uiScale

            ColumnLayout {
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                spacing: 14 * Scaling.uiScale

                // Page header
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    spacing: 4 * Scaling.uiScale

                    Label {
                        text: "Network"
                        font: Scaling.uiHeaderFont
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        elide: Text.ElideNone
                    }

                    Label {
                        text: "Network-related preferences"
                        font: Scaling.uiFont
                        opacity: 0.7
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        elide: Text.ElideNone
                    }
                }

                GroupBox {
                    title: "Prometheus"
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    font: Scaling.uiFont

                    ColumnLayout {
                        spacing: 10 * Scaling.uiScale
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0

                        Label {
                            text: "Prometheus can be used to gather long-term scene and system data from running installations."
                            font: Scaling.uiFont
                            opacity: 0.7
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                            elide: Text.ElideNone
                        }

                        CheckBox {
                            id: enablePrometheusCheckbox
                            text: "Enable metrics endpoint"
                            font: Scaling.uiFont
                            checked: !!AppSettings.enablePrometheusMetrics
                            onToggled: AppSettings.enablePrometheusMetrics = checked

                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                            spacing: 8 * Scaling.uiScale
                            enabled: enablePrometheusCheckbox.checked
                            opacity: enabled ? 1.0 : 0.55

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.minimumWidth: 0
                                spacing: 10 * Scaling.uiScale

                                Label {
                                    text: "Address"
                                    font: Scaling.uiFont
                                    opacity: 0.9
                                    Layout.preferredWidth: 120 * Scaling.uiScale
                                    Layout.minimumWidth: 0
                                    elide: Text.ElideRight
                                }

                                TextField {
                                    id: addressField
                                    font: Scaling.uiFont

                                    Component.onCompleted: {
                                        text = String(AppSettings.prometheusAddress ?? "127.0.0.1:8080");
                                        validate(text, false);
                                    }

                                    placeholderText: "127.0.0.1:8080"

                                    property bool isValid: true

                                    color: isValid ? ThemeColors.text : ThemeColors.red

                                    function validate(address, commit) {
                                        isValid = NetUtils.isValidHostAddress(address);
                                        if (!commit)
                                            return;
                                        if (isValid) {
                                            AppSettings.prometheusAddress = address;
                                        } else {
                                            // revert to last saved value
                                            text = String(AppSettings.prometheusAddress ?? "127.0.0.1:8080");
                                            isValid = NetUtils.isValidHostAddress(text);
                                        }
                                    }

                                    onTextEdited: validate(text, false)
                                    onEditingFinished: validate(text, true)
                                }
                            }
                        }
                    }
                }

                Item {
                    Layout.fillHeight: true
                    Layout.minimumHeight: 0
                }
            }
        }
    }
}
