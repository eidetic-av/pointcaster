import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

RowLayout {
    id: root

    property int pairCount: 0
    property bool canUndo: false

    signal resetRequested
    signal undoRequested
    signal alignRequested

    spacing: Math.round(Scaling.uiScale * 12)

    Item {
        Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
    }

    IconButton {
        text: "Reset"
        tooltip: "Remove current alignment data and start over"
        onClicked: root.resetRequested()
    }

    IconButton {
        text: "Undo last pick"
        tooltip: "Undo the most recent point pick"
        iconSource: FontAwesome.icon("solid/rotate-left")
        enabled: root.canUndo
        onClicked: root.undoRequested()
    }

    Label {
        text: root.pairCount + " pair" + (root.pairCount !== 1 ? "s" : "")
        color: ThemeColors.text
    }

    Item {
        Layout.fillWidth: true
    }

    IconButton {
        text: "Align"
        tooltip: "Finish picking pairs and compute coarse transformation"
        enabled: root.pairCount >= 3
        onClicked: root.alignRequested()
    }

    Item {
        Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
    }
}
