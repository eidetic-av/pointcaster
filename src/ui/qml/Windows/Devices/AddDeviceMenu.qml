import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQml.Models

import Pointcaster 1.0

Menu {
    id: root

    required property var workspace

    width: Math.round(220 * Scaling.uiScale)

    Instantiator {
        model: root.workspace ? root.workspace.addDeviceMenuEntries : []

        delegate: MenuItem {
            required property var modelData

            property bool plugin: modelData.kind === "plugin"
            property bool discovered: modelData.kind === "discovered"

            contentItem: Row {
                Image {
                    visible: discovered
                    height: Math.round(12 * Scaling.uiScale)
                    width: Math.round(24 * Scaling.uiScale)
                    fillMode: Image.PreserveAspectFit
                    source: FontAwesome.icon("solid/ethernet")
                    anchors {
                        verticalCenter: parent.verticalCenter
                    }
                }
                Text {
                    id: entryText
                    text: plugin ? modelData.plugin_name : modelData.label
                    color: ThemeColors.text
                    elide: Text.ElideRight
                    verticalAlignment: Text.AlignVCenter
                }
            }

            background: Rectangle {
                color: hovered ? ThemeColors.mid : ThemeColors.almostdark
            }

            onTriggered: {
                root.close();
                if (plugin) {
                    workspace.addNewDevice(modelData.plugin_name);
                } else if (discovered) {
                    workspace.addNewDevice(modelData.plugin_name, modelData.ip, modelData.id);
                }
            }
        }

        onObjectAdded: (i, o) => root.insertItem(i, o)
        onObjectRemoved: (i, o) => root.removeItem(o)
    }
}
