import QtQuick

// This helps release focus after editing input fields.
// It goes on each top level window.
Item {
    id: root

    anchors.fill: parent
    z: 100000

    property bool keepFocusWithinControl: true

    Item {
        id: focusSink
        objectName: "focusSink"
    }

    function release(): void {
        focusSink.forceActiveFocus(Qt.MouseFocusReason);
    }

    function focusedControl(item: Item): Item {
        if (!root.keepFocusWithinControl)
            return item;

        const owner = item.parent;
        return (owner && owner.contentItem === item) ? owner : item;
    }

    PointHandler {
        id: pressWatcher
        acceptedButtons: Qt.AllButtons

        onActiveChanged: {
            if (!pressWatcher.active)
                return;

            const focusItem = root.Window.activeFocusItem;
            if (!focusItem || focusItem === focusSink)
                return;

            const control = root.focusedControl(focusItem);
            if (control.contains(control.mapFromItem(null, pressWatcher.point.scenePosition)))
                return;

            root.release();
        }
    }
}
