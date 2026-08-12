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

    MouseArea {
        id: pressWatcher
        anchors.fill: parent

        hoverEnabled: false
        acceptedButtons: Qt.AllButtons

        onPressed: mouse => {
            // let the press through to whatever was actually clicked
            mouse.accepted = false;

            const focusItem = root.Window.activeFocusItem;
            if (!focusItem || focusItem === focusSink)
                return;

            const control = root.focusedControl(focusItem);
            if (control.contains(pressWatcher.mapToItem(control, mouse.x, mouse.y)))
                return;

            root.release();
        }
    }
}
