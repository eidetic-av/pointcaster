import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

ToolTip {
    id: root

    property string textValue: ""

    text: textValue
    visible: !!textValue && parent && parent.hovered
    delay: 400

    font: Scaling.uiFont
    opacity: 0.95

    contentItem: Text {
        text: root.text
        font: root.font
        color: ThemeColors.highlightedText
        wrapMode: Text.WordWrap
    }
}
