import Pointcaster 1.0
import Pointcaster.Workspace 1.0
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

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

                AlignmentView {
                    id: secondaryDeviceView
                    // TODO
                    deviceAdapter: workspace && workspace.deviceAdapters.length > 0 ? workspace.deviceAdapters[0] : null
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
            }

            Label {
                text: "Confirmations"
            }
        }
    }
}
