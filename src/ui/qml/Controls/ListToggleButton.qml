import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

// icon-only toggle, used for on/off row controls (active, render,
// broadcasting, etc). dims when "gated" (the toggle is on but something
// upstream overrides it off)
CheckBox {
    id: root

    property url iconOn
    property url iconOff
    property bool gated: false
    property string tip: ""
    property real baseOpacity: 0.6
    property real hoverOpacity: 0.6
    property real offOpacity: baseOpacity

    readonly property bool hasIconOff: String(iconOff).length > 0

    // when off and no iconOff was given, overlay a slash over top
    property bool overlayOffSlash: false
    readonly property bool showOffSlash: overlayOffSlash && !root.checked && !hasIconOff

    readonly property real iconSize: 16 * Scaling.uiScale
    property real iconPadding: 0 * Scaling.uiScale

    Layout.fillHeight: true
    Layout.minimumWidth: Math.round(iconSize + iconPadding * 2)
    Layout.maximumWidth: Math.round(iconSize + iconPadding * 2)

    indicator: Item {
        anchors.centerIn: parent
        width: root.iconSize
        height: root.iconSize

        Image {
            anchors.fill: parent
            source: (root.checked || !root.hasIconOff) ? root.iconOn : root.iconOff
            sourceSize: Qt.size(root.iconSize, root.iconSize)
            opacity: root.gated ? 0.25 : (root.hovered ? root.hoverOpacity : (root.checked ? root.baseOpacity : root.offOpacity))
        }

        Image {
            anchors.fill: parent
            visible: root.showOffSlash
            source: FontAwesome.icon("solid/slash")
            sourceSize: Qt.size(root.iconSize, root.iconSize)
            opacity: root.gated ? 0.25 : (root.hovered ? root.hoverOpacity : root.offOpacity)
        }
    }

    InfoToolTip {
        textValue: root.tip
    }
}
