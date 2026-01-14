import QtQuick
import QtQuick.Controls

import Pointcaster.Workspace 1.0

AbstractButton {
    id: root

    property url iconSource
    property int iconSize: 9
    property int iconTextSpacing: 5
    property int iconLeftPadding: 12

    property color backgroundColor: DarkPalette.middark
    property color hoverColor: DarkPalette.mid
    property color pressedColor: DarkPalette.midlight

    property color borderColor: DarkPalette.mid
    property int borderWidth: 1
    property int radius: 3

    property int contentLeftPadding: 8
    property int contentRightPadding: 8
    property int contentTopPadding: 6
    property int contentBottomPadding: 6

    readonly property bool iconOnly: text.length === 0
    readonly property bool hasIcon: iconSource !== ""

    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    implicitWidth: contentLeftPadding
                   + content.implicitWidth
                   + contentRightPadding
    implicitHeight: contentTopPadding
                    + content.implicitHeight
                    + contentBottomPadding

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

    contentItem: Item {
        id: content

        x: root.contentLeftPadding
        y: root.contentTopPadding

        implicitWidth:
            (icon.visible ? root.iconLeftPadding + icon.width : 0)
            + (label.visible
               ? (icon.visible ? root.iconTextSpacing : 0) + label.implicitWidth
               : 0)

        implicitHeight: Math.max(
            icon.visible ? icon.height : 0,
            label.visible ? label.implicitHeight : 0
        )

        Image {
            id: icon
            visible: root.hasIcon
            source: root.iconSource

            width: root.iconSize
            height: root.iconSize

            anchors.left: parent.left
            anchors.leftMargin: root.iconLeftPadding   // ← applied here
            anchors.verticalCenter: parent.verticalCenter

            fillMode: Image.PreserveAspectFit
            smooth: true
            mipmap: true
        }

        Text {
            id: label
            text: root.text
            visible: !root.iconOnly

            anchors.left: icon.visible ? icon.right : parent.left
            anchors.leftMargin: icon.visible ? root.iconTextSpacing : 0
            anchors.verticalCenter: parent.verticalCenter

            font: root.font
            color: root.palette.buttonText
            elide: Text.ElideRight
        }
    }
}
