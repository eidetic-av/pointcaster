import QtQuick
import QtQuick.Controls

import Pointcaster.Workspace 1.0

AbstractButton {
    id: root

    property url iconSource
    property int iconSize: 9
    property int iconTextSpacing: 5

    property color backgroundColor: DarkPalette.middark
    property color hoverColor: DarkPalette.mid
    property color pressedColor: DarkPalette.midlight

    property color borderColor: DarkPalette.mid
    property int borderWidth: 1
    property int radius: 3

    property string tooltip

    // single source of truth for outer padding
    padding: 0
    leftPadding: 8
    rightPadding: 8
    topPadding: 6
    bottomPadding: 6

    readonly property bool hasIcon: iconSource !== "" && iconSource !== undefined && iconSource !== null
    readonly property bool hasText: text.length > 0

    hoverEnabled: true
    focusPolicy: Qt.StrongFocus
    opacity: enabled ? 1.0 : 0.66

    background: Rectangle {
        anchors.fill: parent
        radius: root.radius

        color: {
            if (!root.enabled) return DarkPalette.middark
            if (root.pressed) return root.pressedColor
            if (root.hovered) return root.hoverColor
            return root.backgroundColor
        }

        border.width: root.borderWidth
        border.color: root.borderColor

        Rectangle {
            visible: root.activeFocus
            anchors.fill: parent
            color: "transparent"
            radius: parent.radius
            border.width: 1
            border.color: DarkPalette.highlight
        }
    }

    // Let AbstractButton size itself from contentItem + paddings.
    contentItem: Item {
        implicitWidth: row.implicitWidth
        implicitHeight: row.implicitHeight

        Row {
            id: row
            anchors.centerIn: parent
            spacing: (root.hasIcon && root.hasText) ? root.iconTextSpacing : 0

            Image {
                visible: root.hasIcon
                source: root.iconSource
                width: root.iconSize
                height: root.hasText ? contentText.height : root.iconSize
                fillMode: Image.PreserveAspectFit
                smooth: true
                mipmap: true
            }

            Text {
                id: contentText
                visible: root.hasText
                text: root.text
                font: root.font
                color: root.palette.buttonText
                elide: Text.ElideRight
                verticalAlignment: Text.AlignVCenter
            }
        }
    }

    ToolTip {
        visible: tooltip ? parent.hovered : false
        text: tooltip
        delay: 400
    }

}
