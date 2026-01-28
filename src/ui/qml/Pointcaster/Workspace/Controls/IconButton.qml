import QtQuick
import QtQuick.Controls

import Pointcaster.Workspace 1.0

AbstractButton {
    id: root

    property url iconSource
    property int iconSize: Math.round(9 * Scaling.uiScale)
    property int iconTextSpacing: Math.round(5 * Scaling.uiScale)

    property color backgroundColor: ThemeColors.middark
    property color hoverColor: ThemeColors.mid
    property color pressedColor: ThemeColors.midlight

    property color borderColor: ThemeColors.mid
    property int borderWidth: 1

    property int radius: Math.round(3 * Scaling.uiScale)

    property real topLeftRadius: -1
    property real topRightRadius: -1
    property real bottomLeftRadius: -1
    property real bottomRightRadius: -1

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

    readonly property real effectiveTopLeftRadius: (root.topLeftRadius >= 0) ? root.topLeftRadius : root.radius
    readonly property real effectiveTopRightRadius: (root.topRightRadius >= 0) ? root.topRightRadius : root.radius
    readonly property real effectiveBottomLeftRadius: (root.bottomLeftRadius >= 0) ? root.bottomLeftRadius : root.radius
    readonly property real effectiveBottomRightRadius: (root.bottomRightRadius >= 0) ? root.bottomRightRadius : root.radius

    background: Rectangle {
        anchors.fill: parent

        radius: root.radius
        topLeftRadius: root.effectiveTopLeftRadius
        topRightRadius: root.effectiveTopRightRadius
        bottomLeftRadius: root.effectiveBottomLeftRadius
        bottomRightRadius: root.effectiveBottomRightRadius

        color: {
            if (!root.enabled)
                return ThemeColors.middark;
            if (root.pressed)
                return root.pressedColor;
            if (root.hovered)
                return root.hoverColor;
            return root.backgroundColor;
        }

        border.width: root.borderWidth
        border.color: root.borderColor

        Rectangle {
            visible: root.activeFocus
            anchors.fill: parent
            color: "transparent"

            radius: root.radius
            topLeftRadius: root.effectiveTopLeftRadius
            topRightRadius: root.effectiveTopRightRadius
            bottomLeftRadius: root.effectiveBottomLeftRadius
            bottomRightRadius: root.effectiveBottomRightRadius

            border.width: root.borderWidth
            border.color: ThemeColors.highlight
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
