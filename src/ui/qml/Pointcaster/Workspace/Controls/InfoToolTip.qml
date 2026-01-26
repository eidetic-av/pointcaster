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
}
