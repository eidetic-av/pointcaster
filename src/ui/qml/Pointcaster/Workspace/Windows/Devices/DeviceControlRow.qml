import QtQml.Models
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Row {
    id: root

    required property var adapter

    spacing: Math.round(8 * Scaling.uiScale)
    enabled: adapter != null

    IconButton {
        id: startStopButton

        font: Scaling.uiFont
        tooltip: "Start/stop selected device"
        text: {
            if (!root.adapter)
                return "Start";
            return root.adapter.status === UiEnums.WorkspaceDeviceStatus.Active ? "Stop" : "Start";
        }
        iconSource: {
            if (!root.adapter)
                return FontAwesome.icon("solid/play");
            return root.adapter.status === UiEnums.WorkspaceDeviceStatus.Active ? FontAwesome.icon("solid/stop") : FontAwesome.icon("solid/play");
        }
        enabled: root.adapter && root.adapter.status && root.adapter.status !== UiEnums.WorkspaceDeviceStatus.Loading
        onClicked: {
            if (!root.adapter)
                return;
            if (root.adapter.status === UiEnums.WorkspaceDeviceStatus.Active)
                root.adapter.stop();
            else
                root.adapter.start();
        }
    }

    IconButton {
        id: restartButton

        font: Scaling.uiFont
        tooltip: "Restart selected device"
        text: "Restart"
        iconSource: FontAwesome.icon("solid/rotate-right")
        enabled: root.adapter && root.adapter.status && root.adapter.status === UiEnums.WorkspaceDeviceStatus.Active
        onClicked: {
            if (root.adapter)
                root.adapter.restart();
        }
    }

    // TODO whats this?
    Item {
        width: 1
        Layout.fillWidth: false
    }
}
