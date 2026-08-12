import QtQuick

import Pointcaster 1.0

// one line of the log console: "[level] message", the level coloured by
// severity and the message wrapped to whatever width it's given, growing the
// row a line at a time rather than eliding

Item {
    id: root

    property string level: ""
    property string message: ""

    readonly property var levelColors: ({
            "trace": ThemeColors.midlight,
            "debug": ThemeColors.blue,
            "info": ThemeColors.text,
            "warning": ThemeColors.yellow,
            "error": ThemeColors.red,
            "critical": ThemeColors.red
        })

    readonly property real singleLineHeight: Math.round(20 * Scaling.uiScale)

    readonly property real textMargin: Math.max(0, Math.round((singleLineHeight - logFontMetrics.height) / 2))

    implicitHeight: textMargin * 2 + messageLabel.contentHeight

    FontMetrics {
        id: logFontMetrics
        font: Scaling.monoFont
    }

    Text {
        id: levelLabel

        anchors.top: parent.top
        anchors.topMargin: root.textMargin

        text: "[" + root.level + "]"
        font: Scaling.monoFont
        color: root.levelColors[root.level] || ThemeColors.text
    }

    Text {
        id: messageLabel

        anchors.left: levelLabel.right
        anchors.leftMargin: Math.round(6 * Scaling.uiScale)
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.topMargin: root.textMargin

        text: root.message
        wrapMode: Text.WrapAtWordBoundaryOrAnywhere
        font: Scaling.monoFont
        color: ThemeColors.text
    }
}
