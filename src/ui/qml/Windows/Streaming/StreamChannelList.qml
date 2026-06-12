import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Rectangle {
    id: root

    property var workspace: null

    readonly property real rowHeight: Math.max(Math.round(26 * Scaling.uiScale), Math.ceil(Scaling.pointSize * 2))

    color: ThemeColors.alternateBase
    border.color: ThemeColors.mid
    border.width: 1
    clip: true

    ListView {
        id: list
        anchors.fill: parent
        anchors.margins: 1
        clip: true

        model: root.workspace ? root.workspace.streamChannels : null

        ScrollBar.vertical: ScrollBar {
            policy: ScrollBar.AsNeeded
        }

        delegate: Rectangle {
            id: channelRow

            required property int index
            required property string address
            required property bool enabled
            required property int status

            width: list.width
            height: root.rowHeight

            readonly property bool selected: root.workspace && root.workspace.selectedStreamChannelIndex === channelRow.index

            color: hoverHandler.hovered ? ThemeColors.midlight : channelRow.selected ? ThemeColors.mid : ThemeColors.almostdark

            HoverHandler {
                id: hoverHandler
            }

            TapHandler {
                onTapped: root.workspace.selectedStreamChannelIndex = channelRow.index
            }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: Math.round(8 * Scaling.uiScale)
                anchors.rightMargin: Math.round(8 * Scaling.uiScale)
                spacing: Math.round(8 * Scaling.uiScale)

                Rectangle {
                    id: statusDot
                    Layout.alignment: Qt.AlignVCenter
                    width: Math.round(6 * Scaling.uiScale)
                    height: width
                    radius: width / 2
                    color: {
                        switch (channelRow.status) {
                        case Enum.StreamChannelStatus.Connected:
                            return ThemeColors.green;
                        case Enum.StreamChannelStatus.Live:
                            return ThemeColors.blue;
                        default:
                            return ThemeColors.inactive;
                        }
                    }
                }

                Text {
                    Layout.fillWidth: true
                    text: channelRow.address
                    color: ThemeColors.text
                    font: Scaling.uiFont
                    elide: Text.ElideRight
                }

                ListToggleButton {
                    iconOn: FontAwesome.icon("solid/users")
                    iconOff: FontAwesome.icon("solid/users-slash")
                    baseOpacity: 0.7
                    hoverOpacity: 0.9
                    offOpacity: 0.25
                    tip: "Toggle broadcasting on this channel"
                    checked: channelRow.enabled
                    onToggled: root.workspace.streamChannels.setChannelEnabled(channelRow.index, checked)
                }
            }
        }
    }
}
