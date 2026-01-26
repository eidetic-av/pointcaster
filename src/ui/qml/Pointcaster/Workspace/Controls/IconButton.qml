import QtQuick
import QtQuick.Controls

import Pointcaster.Workspace 1.0

AbstractButton {
    id: root

    property url iconSource
    property int iconSize: Math.round(9 * Scaling.uiScale)
    property int iconTextSpacing: Math.round(5 * Scaling.uiScale)

    property color backgroundColor: DarkPalette.middark
    property color hoverColor: DarkPalette.mid
    property color pressedColor: DarkPalette.midlight

    property color borderColor: DarkPalette.mid

    property string tooltip

    font: Scaling.uiFont

    padding: 0
    leftPadding: Math.round(8 * Scaling.uiScale)
    rightPadding: Math.round(8 * Scaling.uiScale)
    topPadding: Math.round(6 * Scaling.uiScale)
    bottomPadding: Math.round(6 * Scaling.uiScale)

    readonly property bool hasIcon: iconSource !== "" && iconSource !== undefined && iconSource !== null
    readonly property bool hasText: text.length > 0

    hoverEnabled: true
    focusPolicy: Qt.StrongFocus
    opacity: enabled ? 1.0 : 0.66

    background: Rectangle {
        anchors.fill: parent
        radius: Math.round(3 * Scaling.uiScale)

        color: {
            if (!root.enabled)
                return DarkPalette.middark;
            if (root.pressed)
                return root.pressedColor;
            if (root.hovered)
                return root.hoverColor;
            return root.backgroundColor;
        }

        border.width: 1
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

    InfoToolTip {
        textValue: tooltip
    }
}
