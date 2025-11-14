import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Pointcaster.Controls

ApplicationWindow {
    id: root
    width: 1280
    height: 800
    visible: true
    title: "Material Layout Demo"

    readonly property int navRailWidth: 72
    readonly property int detailPaneWidth: 400
    readonly property int cardGridMinWidth: 720

    minimumWidth: navRailWidth + detailPaneWidth + cardGridMinWidth
    minimumHeight: 500

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // Left navigation rail
        Rectangle {
            Layout.preferredWidth: root.navRailWidth
            Layout.minimumWidth: root.navRailWidth
            Layout.maximumWidth: root.navRailWidth
            Layout.fillHeight: true

            Column {
                anchors.centerIn: parent
                spacing: 24

                ToolButton { icon.name: "search" }
                ToolButton { icon.name: "favorite" }
                ToolButton { icon.name: "settings" }
            }
        }

        // Center scrollable card grid (wrapped in a ScrollView for proper Layout behavior)
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumWidth: root.cardGridMinWidth
            clip: true

            GridLayout {
                id: grid
                columns: 3
                columnSpacing: 16
                rowSpacing: 16
                anchors.margins: 32

                Repeater {
                    model: 9
                    Rectangle {
                        Layout.preferredWidth: 240
                        Layout.preferredHeight: 300
                        radius: 12
                        color: "#ffffff"
                        border.color: "#dddddd"

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 8

                            Rectangle {
                                Layout.preferredHeight: 120
                                Layout.fillWidth: true
                                radius: 8
                                color: "#ddd"
                                Text {
                                    anchors.centerIn: parent
                                    text: "Image"
                                    font.pointSize: 10
                                }
                            }

                            Text {
                                text: "Product Title"
                                font.bold: true
                                font.pointSize: 14
                                wrapMode: Text.WordWrap
                            }

                            Text {
                                text: "From your recent favorites"
                                font.pointSize: 12
                                color: "#666"
                                wrapMode: Text.WordWrap
                            }

                            TextButton {
                                text: "View"
                                Layout.alignment: Qt.AlignHCenter
                                Layout.preferredWidth: 120
                            }
                        }
                    }
                }
            }
        }

        // Right detail pane
        Rectangle {
            Layout.preferredWidth: root.detailPaneWidth
            Layout.minimumWidth: root.detailPaneWidth
            Layout.maximumWidth: root.detailPaneWidth
            Layout.fillHeight: true
            color: "#f5f5f5"
            border.color: "#cccccc"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                Text {
                    text: "Details"
                    font.pixelSize: 20
                    font.bold: true
                }

                Text {
                    text: "This is the detail panel. Select a product to see more information here."
                    wrapMode: Text.WordWrap
                    font.pixelSize: 14
                }

                TextButton {
                    text: "Add to Basket"
                    Layout.alignment: Qt.AlignLeft
                }
            }
        }
    }
}
