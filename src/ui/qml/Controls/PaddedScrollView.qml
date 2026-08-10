import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

// A ScrollView whose vertical scrollbar shows when needed and sits in its own
// gutter rather than over the content

ScrollView {
    id: root

    property int scrollBarGap: Math.round(6 * Scaling.uiScale)

    property bool scrollBarEnabled: true

    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

    ScrollBar.vertical: ScrollBar {
        id: verticalScrollBar

        parent: root
        x: root.mirrored ? 0 : root.width - width
        y: root.topPadding
        height: root.availableHeight

        policy: (root.scrollBarEnabled && size < 1.0) ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff

        leftPadding: root.scrollBarGap
        hoverEnabled: true

        contentItem: Rectangle {
            implicitWidth: Math.round(6 * Scaling.uiScale)
            radius: width / 2
            color: verticalScrollBar.pressed ? ThemeColors.highlight : verticalScrollBar.hovered ? ThemeColors.midlight : ThemeColors.mid
        }
    }

    // with the bar hidden its width is not reserved
    rightPadding: padding + ((ScrollBar.vertical.visible && ScrollBar.vertical.interactive) ? ScrollBar.vertical.width : 0)

    Binding {
        target: root.contentItem
        property: "boundsBehavior"
        value: Flickable.StopAtBounds
    }
}
