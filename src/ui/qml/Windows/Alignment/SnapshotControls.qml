import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

RowLayout {
    id: root

    property var deviceAdapters: []
    property int primaryDeviceIndex: 0
    property int secondaryDeviceIndex: 1

    signal primaryDeviceChanged(int index)
    signal secondaryDeviceChanged(int index)
    signal snapshotRequested()

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
            textRole: "id"

            contentItem: Text {
                text: primaryDeviceList.displayText
                font: Scaling.uiFont
                color: ThemeColors.text
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }

            delegate: ItemDelegate {
                width: primaryDeviceList.width
                text: modelData.id
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
            textRole: "id"

            contentItem: Text {
                text: secondaryDeviceList.displayText
                font: Scaling.uiFont
                color: ThemeColors.text
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }

            delegate: ItemDelegate {
                width: secondaryDeviceList.width
                text: modelData.id
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