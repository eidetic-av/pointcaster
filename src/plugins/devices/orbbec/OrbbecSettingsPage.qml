import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

Item {
    id: root

    implicitWidth: Math.round(520 * Scaling.uiScale)
    implicitHeight: Math.round(360 * Scaling.uiScale)

    readonly property string sdkVersionKey: "plugins/orbbec/sdkVersion"

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
                    text: "Orbbec"
                    font: Scaling.uiHeaderFont
                    elide: Text.ElideNone
                }

                Label {
                    text: "Options for the Orbbec device plugin."
                    font: Scaling.uiFont
                    opacity: 0.7
                    wrapMode: Text.WordWrap
                    elide: Text.ElideNone
                }
            }

            GroupBox {
                id: sdkOptions
                title: "SDK"
                anchors.left: parent.left
                anchors.right: parent.right

                Column {
                    width: parent.width
                    spacing: Math.round(10 * Scaling.uiScale)

                    Label {
                        text: "Switching SDK version requires a restart of Pointcaster."
                        font: Scaling.uiFont
                        opacity: 0.7
                        wrapMode: Text.WordWrap
                        elide: Text.ElideNone
                        width: parent.width - Math.round(5 * Scaling.uiScale)
                    }

                    Row {
                        spacing: Math.round(10 * Scaling.uiScale)

                        Label {
                            text: "SDK version"
                            font: Scaling.uiFont
                            opacity: 0.9
                            width: 120 * Scaling.uiScale
                            anchors.verticalCenter: parent.verticalCenter
                        }

                        ComboBox {
                            id: sdkVersionCombo
                            font: Scaling.uiFont
                            textRole: "text"
                            valueRole: "value"
                            width: 220 * Scaling.uiScale

                            model: [
                                {
                                    text: "Orbbec SDK v2",
                                    value: 2
                                },
                                {
                                    text: "Orbbec SDK v1",
                                    value: 1
                                }
                            ]

                            Component.onCompleted: {
                                const saved = Number(AppSettings.value(root.sdkVersionKey, 2));
                                const idx = indexOfValue(saved === 1 ? 1 : 2);
                                if (idx >= 0)
                                    currentIndex = idx;
                            }

                            onActivated: {
                                AppSettings.setValue(root.sdkVersionKey, currentValue);
                                AppSettings.sync();
                            }
                        }
                    }
                }
            }
        }
    }
}
