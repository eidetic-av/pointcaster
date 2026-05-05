import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

KDDW.DockWidget {
    id: root
    uniqueName: "alignmentWindow"
    title: "Alignment"

    property var workspace: null

    Item {
        anchors.fill: parent

        Rectangle {
            anchors.fill: parent
            color: ThemeColors.base
        }

        ColumnLayout {
            anchors.fill: parent

            RowLayout {

                Label {
                    text: "Alignment"
                }

                Button {
                    text: "Snapshot"
                    onClicked: {
                        primaryDeviceView.snapshot();
                        secondaryDeviceView.snapshot();
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true

                AlignmentView {
                    id: primaryDeviceView
                    deviceAdapter: workspace && workspace.deviceAdapters.length > 0 ? workspace.deviceAdapters[0] : null
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                Rectangle {
                    width: 1
                    Layout.fillHeight: true
                    color: ThemeColors.middark
                }

                Item {
                    id: secondaryDeviceContainer
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    AlignmentView {
                        id: secondaryDeviceView
                        visible: workspace.deviceAdapters.length > 1
                        deviceAdapter: workspace && workspace.deviceAdapters.length > 1 ? workspace.deviceAdapters[1] : null
                        anchors.fill: parent
                    }

                    Label {
                        anchors.fill: parent
                        visible: workspace.deviceAdapters.length < 2
                        text: "Alignment requires 2 or more devices"
                        background: Rectangle {
                            color: ThemeColors.dark
                        }
                    }
                }
            }

            Label {
                text: "Confirmations"
            }
        }
    }
}
