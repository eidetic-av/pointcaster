// lives in kddw's indicator window, which runs its own qml engine. that engine
// gets qrc:/qt/qml added to its import path in tab_drop_indicators.cc so the
// Pointcaster module resolves here. unit stands in for Scaling.uiScale, since
// the tile is sized from c++ to match whatever kddw gives the indicator item
import QtQuick
import com.kdab.dockwidgets 2.0
import Pointcaster 1.0

Rectangle {
    id: root

    readonly property int dropLocation: parent ? parent.indicatorType : KDDockWidgets.DropLocation_None
    readonly property bool active: parent ? parent.isHovered : false

    readonly property real unit: width / 40
    readonly property real hairline: Math.max(1, Math.round(unit))

    readonly property color accent: ThemeColors.highlight
    readonly property color plate: ThemeColors.almostdark
    readonly property color plateBorder: ThemeColors.mid
    readonly property color glyphColour: ThemeColors.midlight

    readonly property bool isCentre: dropLocation === KDDockWidgets.DropLocation_Center

    readonly property bool isOuter: dropLocation === KDDockWidgets.DropLocation_OutterLeft
                                    || dropLocation === KDDockWidgets.DropLocation_OutterRight
                                    || dropLocation === KDDockWidgets.DropLocation_OutterTop
                                    || dropLocation === KDDockWidgets.DropLocation_OutterBottom

    // the half of the frame a drop would take, in glyph-local design units
    readonly property rect fillArea: {
        switch (dropLocation) {
        case KDDockWidgets.DropLocation_Left:
        case KDDockWidgets.DropLocation_OutterLeft:
            return Qt.rect(1, 1, 12, 20);
        case KDDockWidgets.DropLocation_Right:
        case KDDockWidgets.DropLocation_OutterRight:
            return Qt.rect(13, 1, 12, 20);
        case KDDockWidgets.DropLocation_Top:
        case KDDockWidgets.DropLocation_OutterTop:
            return Qt.rect(1, 1, 24, 10);
        case KDDockWidgets.DropLocation_Bottom:
        case KDDockWidgets.DropLocation_OutterBottom:
            return Qt.rect(1, 11, 24, 10);
        default:
            return Qt.rect(0, 0, 0, 0);
        }
    }

    color: active ? Qt.rgba(accent.r, accent.g, accent.b, 0.12) : plate
    radius: Math.round(3 * unit)
    border.width: hairline
    border.color: active ? accent : plateBorder

    Behavior on color {
        ColorAnimation {
            duration: Math.round(90 * root.unit)
            easing.type: Easing.InCubic
        }
    }

    Behavior on border.color {
        ColorAnimation {
            duration: Math.round(90 * root.unit)
            easing.type: Easing.InCubic
        }
    }

    // outer targets dock against the window rather than the hovered group
    Rectangle {
        anchors.fill: parent
        anchors.margins: Math.round(3 * root.unit)
        visible: root.isOuter
        color: "transparent"
        radius: Math.round(2 * root.unit)
        border.width: root.hairline
        border.color: root.active ? Qt.rgba(root.accent.r, root.accent.g, root.accent.b, 0.55)
                                  : root.plateBorder
    }

    Item {
        id: glyph

        x: Math.round(7 * root.unit)
        y: Math.round(9 * root.unit)
        width: Math.round(26 * root.unit)
        height: Math.round(22 * root.unit)

        Rectangle {
            anchors.fill: parent
            color: "transparent"
            border.width: root.hairline
            border.color: root.active ? root.accent : root.glyphColour
        }

        Rectangle {
            visible: !root.isCentre
            x: Math.round(root.fillArea.x * root.unit)
            y: Math.round(root.fillArea.y * root.unit)
            width: Math.round(root.fillArea.width * root.unit)
            height: Math.round(root.fillArea.height * root.unit)
            color: root.active ? root.accent : root.glyphColour
        }

        Item {
            anchors.fill: parent
            visible: root.isCentre

            Rectangle {
                x: Math.round(1 * root.unit)
                y: Math.round(1 * root.unit)
                width: Math.round(12 * root.unit)
                height: Math.round(5 * root.unit)
                color: root.glyphColour
                opacity: 0.65
            }

            Rectangle {
                x: Math.round(1 * root.unit)
                y: Math.round(6 * root.unit)
                width: Math.round(24 * root.unit)
                height: Math.round(15 * root.unit)
                color: root.glyphColour
                opacity: 0.3
            }

            Rectangle {
                x: Math.round(13 * root.unit)
                y: Math.round(1 * root.unit)
                width: Math.round(12 * root.unit)
                height: Math.round(5 * root.unit)
                color: root.active ? root.accent : root.glyphColour
            }
        }
    }
}
