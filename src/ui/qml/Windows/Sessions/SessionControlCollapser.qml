import QtQuick
import Pointcaster 1.0

Item {
    id: root

    required property Item contentItem
    required property int direction

    enum Direction {
        CollapseUp,
        CollapseDown,
        CollapseLeft,
        CollapseRight
    }

    property bool collapsed: false
    property int buttonAlignment: Qt.AlignRight

    property real openAmount: collapsed ? 0 : 1

    Behavior on openAmount {
        NumberAnimation {
            duration: 160
            easing.type: Easing.OutCubic
        }
    }

    property alias container: contentPanel

    readonly property bool _isVertical: direction === SessionControlCollapser.CollapseUp || direction === SessionControlCollapser.CollapseDown
    readonly property bool _buttonOnLeft: buttonAlignment === Qt.AlignLeft

    readonly property real _margin: Math.round(4 * Scaling.uiScale)
    readonly property real _buttonLong: Math.round(16 * Scaling.uiScale)
    readonly property real _buttonShort: Math.round(11 * Scaling.uiScale)

    implicitWidth: _isVertical ? contentPanel.width : Math.max(contentPanel.width, toggleCollapsedButton.width)
    implicitHeight: contentPanel.height + toggleCollapsedButton.height

    onContentItemChanged: {
        if (contentItem) {
            contentItem.parent = contentPanel;
            contentItem.anchors.centerIn = contentPanel;
        }
    }

    Rectangle {
        id: contentPanel
        color: ThemeColors.dark
        opacity: 0.75
        clip: true

        property real spacing: Math.round(6 * Scaling.uiScale)

        width: root._isVertical ? (root.contentItem.width + spacing) : (root.contentItem.width + spacing) * root.openAmount
        height: root._isVertical ? (root.contentItem.height + spacing) * root.openAmount : (root.contentItem.height + spacing)

        x: root.direction === SessionControlCollapser.CollapseRight ? root.width - width : 0
        y: root.direction === SessionControlCollapser.CollapseUp ? 0 : toggleCollapsedButton.height
    }

    IconButton {
        id: toggleCollapsedButton

        width: root._isVertical ? root._buttonLong : root._buttonShort
        implicitHeight: root._isVertical ? root._buttonShort : root._buttonLong

        x: root._isVertical ? (root._buttonOnLeft ? contentPanel.x + root._margin : contentPanel.x + contentPanel.width - width - root._margin) : (root.direction === SessionControlCollapser.CollapseLeft ? 0 : root.width - width)
        y: root.direction === SessionControlCollapser.CollapseUp ? contentPanel.y + contentPanel.height : 0

        iconSource: {
            const d = root.direction;
            const c = root.collapsed;
            const name = d === SessionControlCollapser.CollapseUp ? (c ? "down" : "up") : d === SessionControlCollapser.CollapseDown ? (c ? "up" : "down") : d === SessionControlCollapser.CollapseLeft ? (c ? "right" : "left") : (c ? "left" : "right");
            return FontAwesome.icon("solid/caret-" + name);
        }

        iconSize: Math.round(9 * Scaling.uiScale)
        iconColor: !pressed ? ThemeColors.mid : ThemeColors.midlight
        opacity: 0.75

        leftPadding: root._isVertical ? Math.round(3 * Scaling.uiScale) : 0
        rightPadding: root._isVertical ? Math.round(3 * Scaling.uiScale) : 0
        topPadding: root._isVertical ? 0 : Math.round(3 * Scaling.uiScale)
        bottomPadding: root._isVertical ? 0 : Math.round(3 * Scaling.uiScale)

        topLeftRadius: (root.direction === SessionControlCollapser.CollapseUp || root.direction === SessionControlCollapser.CollapseLeft) ? 0 : undefined
        topRightRadius: (root.direction === SessionControlCollapser.CollapseUp || root.direction === SessionControlCollapser.CollapseRight) ? 0 : undefined
        bottomLeftRadius: (root.direction === SessionControlCollapser.CollapseDown || root.direction === SessionControlCollapser.CollapseLeft) ? 0 : undefined
        bottomRightRadius: (root.direction === SessionControlCollapser.CollapseDown || root.direction === SessionControlCollapser.CollapseRight) ? 0 : undefined

        backgroundColor: ThemeColors.dark
        hoverColor: ThemeColors.middark
        pressedColor: ThemeColors.mid
        borderWidth: 0
        onClicked: root.collapsed = !root.collapsed
    }
}
