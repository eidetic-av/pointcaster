import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Item {
    id: root

    implicitWidth: Math.round(520 * Scaling.uiScale)
    implicitHeight: Math.round(360 * Scaling.uiScale)

    ScrollView {
        id: scroll
        anchors.fill: parent
        clip: true

        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        Column {
            width: parent.width
            spacing: 14 * Scaling.uiScale

            Column {
                id: pageHeading
                width: parent.width
                spacing: 4 * Scaling.uiScale

                Label {
                    text: "Metrics"
                    font: Scaling.uiHeaderFont
                    elide: Text.ElideNone
                }

                Label {
                    text: "Settings for gathering session telemtry and performance data."
                    font: Scaling.uiFont
                    opacity: 0.7
                    wrapMode: Text.WordWrap
                    elide: Text.ElideNone
                }
            }

            GroupBox {
                id: prometheusOptions
                title: "Prometheus"
                anchors.left: parent.left
                anchors.right: parent.right

                Column {
                    width: parent.width
                    spacing: Math.round(10 * Scaling.uiScale)

                    Label {
                        text: "Prometheus can be used to gather time series data from running installations."
                        font: Scaling.uiFont
                        opacity: 0.7
                        wrapMode: Text.WordWrap
                        elide: Text.ElideNone
                        width: parent.width - Math.round(5 * Scaling.uiScale)
                    }

                    CheckBox {
                        id: enablePrometheusCheckbox
                        text: "Enable metrics endpoint"
                        font: Scaling.uiFont
                        checked: !!AppSettings.enablePrometheusMetrics
                        onToggled: AppSettings.enablePrometheusMetrics = checked
                    }

                    Row {
                        spacing: Math.round(10 * Scaling.uiScale)

                        Label {
                            text: "Address"
                            font: Scaling.uiFont
                            opacity: 0.9
                            width: 120 * Scaling.uiScale
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

            GroupBox {
                id: tracyOptions
                title: "Tracy"
                anchors.left: parent.left
                anchors.right: parent.right

                Column {
                    width: parent.width
                    spacing: Math.round(10 * Scaling.uiScale)

                    Label {
                        text: "Tracy can profile performance and help locate slowdowns in a Pointcaster workspace."
                        width: parent.width - Math.round(10 * Scaling.uiScale)
                        font: Scaling.uiFont
                        opacity: 0.7
                        wrapMode: Text.WordWrap
                        elide: Text.ElideNone
                    }

                    CheckBox {
                        id: enableTracyCheckbox
                        text: "Enable profiling endpoint (requires restart)"
                        font: Scaling.uiFont
                        checked: !!AppSettings.enableTracyProfiling
                        onToggled: AppSettings.enableTracyProfiling = checked
                    }
                }
            }

            //         Column {
            //             id: prometheusOptions

            //             Label {
            //                 text: "Prometheus"
            //                 font: Scaling.uiFont
            //                 elide: Text.ElideNone
            //             }

            //             GroupBox {
            //                 width: settingsPane.contentWidth
            //                 font: Scaling.uiFont

            //                 ColumnLayout {
            //                     Layout.fillWidth: true
            //                     Layout.minimumWidth: 0
            //                     spacing: 8 * Scaling.uiScale
            //                     enabled: enablePrometheusCheckbox.checked
            //                     opacity: enabled ? 1.0 : 0.55

            //                     RowLayout {
            //                 }
            //             }
            //         }
            //     }

        }

        // Pane {
        //     id: settingsPane
        //     width: scroll.width
        //     padding: 16 * Scaling.uiScale

        //     Column {
        //         width: settingsPane.contentWidth

        //     // ColumnLayout {
        //     //     Layout.fillWidth: true
        //     //     Layout.minimumWidth: 0
        //     //     spacing: 14 * Scaling.uiScale

        //     //     // Page header
        //     //     ColumnLayout {
        //     //         Layout.fillWidth: true
        //     //         Layout.minimumWidth: 0
        //     //         spacing: 4 * Scaling.uiScale

        //     //         Label {
        //     //             text: "Metrics"
        //     //             font: Scaling.uiHeaderFont
        //     //             Layout.fillWidth: true
        //     //             Layout.minimumWidth: 0
        //     //             elide: Text.ElideNone
        //     //         }

        //     //         Label {
        //     //             text: "Settings for gathering session telemtry and performance data."
        //     //             font: Scaling.uiFont
        //     //             opacity: 0.7
        //     //             wrapMode: Text.WordWrap
        //     //             Layout.fillWidth: true
        //     //             Layout.minimumWidth: 0
        //     //             elide: Text.ElideNone
        //     //         }
        //     //     }

        //     //     GroupBox {
        //     //         title: "Prometheus"
        //     //         Layout.fillWidth: true
        //     //         Layout.minimumWidth: 0
        //     //         font: Scaling.uiFont

        //     //         ColumnLayout {
        //     //             spacing: 10 * Scaling.uiScale
        //     //             Layout.fillWidth: true
        //     //             Layout.minimumWidth: 0

        //     //             Rectangle {
        //     //                 width: parent.width
        //     //                 height: 200
        //     //                 color: "blue"
        //     //             }

        //     //     GroupBox {
        //     //         title: "Tracy"
        //     //         Layout.fillWidth: true
        //     //         Layout.minimumWidth: 0
        //     //         font: Scaling.uiFont

        //     //         ColumnLayout {
        //     //             spacing: 10 * Scaling.uiScale
        //     //             Layout.fillWidth: true
        //     //             Layout.minimumWidth: 0

        //     //             Label {
        //     //                 text: "Tracy can be used to profile Pointcaster workspaces, gather frame times, and help track down performance issues within the session pipeline."
        //     //                 font: Scaling.uiFont
        //     //                 opacity: 0.7
        //     //                 wrapMode: Text.WordWrap
        //     //                 Layout.fillWidth: true
        //     //                 elide: Text.ElideNone
        //     //             }

        //     //             CheckBox {
        //     //                 id: enableTracyCheckbox
        //     //                 text: "Enable profiling endpoint"
        //     //                 font: Scaling.uiFont
        //     //                 checked: !!AppSettings.enableTracyProfiling
        //     //                 onToggled: AppSettings.enableTracyProfiling = checked

        //     //                 Layout.fillWidth: true
        //     //                 Layout.minimumWidth: 0
        //     //             }
        //     //         }
        //     //     }

        //     //     Item {
        //     //         Layout.fillHeight: true
        //     //         Layout.minimumHeight: 0
        //     //     }
        //     // }
        // }
    }
}
