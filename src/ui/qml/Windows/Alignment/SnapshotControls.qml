import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

RowLayout {
    id: root

    property var workspace: null
    property var deviceAdapters: workspace.deviceAdapters
    property int primaryDeviceIndex: 0
    property int secondaryDeviceIndex: 1

    signal primaryDeviceChanged(int index)
    signal secondaryDeviceChanged(int index)
    signal snapshotRequested()

    // builds a grandparent/parent/label style path for a device,
    // prefixing with the label or id of each parent group
    function devicePath(adapter) {
        if (!adapter)
            return "";
        const ownLabel = adapter.label && adapter.label.length > 0 ? adapter.label : adapter.id;
        const prefix = devicePrefix(adapter.id);
        return prefix.length > 0 ? prefix + "/" + ownLabel : ownLabel;
    }

    function devicePrefix(nodeId) {
        const rows = root.workspace ? root.workspace.deviceTreeRows : [];
        const byId = {};
        for (let i = 0; i < rows.length; i++)
            byId[rows[i].id] = rows[i];

        const parts = [];
        const current = byId[nodeId];
        let parentId = current ? current.parentId : "";
        while (parentId && parentId.length > 0) {
            const parentRow = byId[parentId];
            if (!parentRow)
                break;
            parts.unshift(parentRow.label && parentRow.label.length > 0 ? parentRow.label : parentRow.id);
            parentId = parentRow.parentId;
        }
        return parts.join("/");
    }

    spacing: Math.round(Scaling.uiScale * 16)

    Item {
        Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
    }

    Column {
        spacing: Math.round(Scaling.uiScale * 4)

        Label {
            text: "Primary Device"
        }

        ComboBox {
            id: primaryDeviceList
            flat: true
            model: root.deviceAdapters
            currentIndex: root.primaryDeviceIndex

            contentItem: Text {
                text: root.devicePath(root.deviceAdapters[primaryDeviceList.currentIndex])
                font: Scaling.uiFont
                color: ThemeColors.text
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }

            delegate: ItemDelegate {
                width: primaryDeviceList.width
                text: root.devicePath(modelData)
                font: Scaling.uiFont
                highlighted: primaryDeviceList.highlightedIndex === index
                enabled: secondaryDeviceList.currentIndex !== index
            }

            onActivated: root.primaryDeviceChanged(currentIndex)
        }
    }

    Column {
        spacing: Math.round(Scaling.uiScale * 4)

        Label {
            text: "Secondary Device"
        }

        ComboBox {
            id: secondaryDeviceList
            flat: true
            model: root.deviceAdapters
            currentIndex: root.secondaryDeviceIndex

            contentItem: Text {
                text: root.devicePath(root.deviceAdapters[secondaryDeviceList.currentIndex])
                font: Scaling.uiFont
                color: ThemeColors.text
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }

            delegate: ItemDelegate {
                width: secondaryDeviceList.width
                text: root.devicePath(modelData)
                font: Scaling.uiFont
                highlighted: secondaryDeviceList.highlightedIndex === index
                enabled: primaryDeviceList.currentIndex !== index
            }

            onActivated: root.secondaryDeviceChanged(currentIndex)
        }
    }

    Item {
        Layout.fillWidth: true
    }

    IconButton {
        text: "Snapshot"
        tooltip: "Capture a frame from the target devices and begin alignment"
        onClicked: root.snapshotRequested()
    }

    Item {
        Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
    }
}