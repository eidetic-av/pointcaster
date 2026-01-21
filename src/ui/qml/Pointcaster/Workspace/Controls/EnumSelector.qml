import QtQuick
import QtQuick.Controls

import Pointcaster 1.0

ComboBox {
    id: root

    required property var options
    // backing enum int
    property int boundValue: 0

    flat: true

    background: Rectangle {
        implicitWidth: parent.width
        implicitHeight: parent.height
        color: (root.activeFocus || rootHover.hovered) ? DarkPalette.almostdark : "transparent"
        radius: 0
        border {
            width: (root.activeFocus || root.visualFocus) ? 1 : 0
            color: DarkPalette.highlight
        }
    }

    contentItem: Text {
        topPadding: 6
        leftPadding: 6
        rightPadding: 6
        bottomPadding: 6
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight

        text: root.displayText
        color: DarkPalette.text
        font: root.font
        opacity: root.enabled ? 1.0 : 0.66
    }

    signal commitValue(int value)

    // prevents some feedback loops with config -> ui -> config -> ui when a value is set
    property bool _syncGuard: false

    model: options || []
    textRole: "text"

    HoverHandler {
        id: rootHover
        cursorShape: Qt.ArrowCursor
    }

    function indexForValue(enumValue): int {
        const m = root.model;
        for (let i = 0; i < m.length; ++i) {
            if (m[i].value === enumValue)
                return i;
        }
        return -1;
    }

    function valueForIndex(index): int {
        const m = root.model;
        if (index < 0 || index >= m.length)
            return boundValue;
        return m[index].value;
    }

    function resyncFromBoundValue(): void {
        if (_syncGuard)
            return;

        const idx = indexForValue(boundValue);
        if (idx >= 0 && currentIndex !== idx) {
            _syncGuard = true;
            currentIndex = idx;
            _syncGuard = false;
        }
    }

    onBoundValueChanged: resyncFromBoundValue()
    onModelChanged: resyncFromBoundValue()
    Component.onCompleted: resyncFromBoundValue()

    onActivated: function (index) {
        if (_syncGuard)
            return;

        const newValue = valueForIndex(index);
        if (boundValue !== newValue) {
            _syncGuard = true;
            boundValue = newValue;
            _syncGuard = false;
        }
        commitValue(newValue);
    }
}
