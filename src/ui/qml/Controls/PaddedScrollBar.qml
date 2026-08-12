import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

ScrollBar {
    id: root

    property int gap: Math.round(6 * Scaling.uiScale)

    property bool barEnabled: true

    policy: (root.barEnabled && size < 1.0) ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff

    leftPadding: root.gap
    hoverEnabled: true

    contentItem: Rectangle {
        implicitWidth: Math.round(6 * Scaling.uiScale)
        radius: width / 2
        color: root.pressed ? ThemeColors.highlight : root.hovered ? ThemeColors.mid : ThemeColors.middark
    }
}
