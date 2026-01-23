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

        // Never allow horizontal scrolling.
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        // Keep content width locked to the viewport width so children can't expand it.
        contentWidth: availableWidth

        Pane {
            // Use contentWidth (locked) rather than availableWidth (can differ in some styles)
            width: scroll.contentWidth
            padding: 16

            ColumnLayout {
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                spacing: 14

                // Page header
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    spacing: 4

                    Label {
                        text: "Network"
                        font.pixelSize: 18
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        elide: Text.ElideNone
                    }

                    Label {
                        text: "Network-related preferences"
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

                    ColumnLayout {
                        spacing: 10
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0

                        Label {
                            text: "Prometheus can be used to gather long-term scene and system data from running installations."
                            opacity: 0.7
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                            elide: Text.ElideNone
                        }

                        CheckBox {
                            id: enablePrometheusCheckbox
                            text: "Enable metrics endpoint"
                            checked: !!AppSettings.enablePrometheusMetrics
                            onToggled: AppSettings.enablePrometheusMetrics = checked

                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                            spacing: 8
                            enabled: enablePrometheusCheckbox.checked
                            opacity: enabled ? 1.0 : 0.55

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.minimumWidth: 0
                                spacing: 10

                                Label {
                                    text: "Address"
                                    opacity: 0.9
                                    Layout.preferredWidth: 120
                                    Layout.minimumWidth: 0
                                    elide: Text.ElideRight
                                }

                                TextField {
                                    id: addressField

                                    Component.onCompleted: {
                                        text = String(AppSettings.prometheusAddress ?? "127.0.0.1:8080");
                                        validate(text, false);
                                    }

                                    placeholderText: "127.0.0.1:8080"

                                    property bool isValid: true

                                    color: isValid ? DarkPalette.text : DarkPalette.red

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
