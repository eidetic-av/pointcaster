import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

KDDW.DockWidget {
    id: root
    uniqueName: "publishersWindow"
    title: "Publishers"

    property var workspace: null

    Item {
        anchors.fill: parent

        Rectangle {
            anchors.fill: parent
            color: ThemeColors.base
        }

        ColumnLayout {
            id: innerContent

            property var kddockwidgets_min_size: Qt.size(Math.round(225 * Scaling.uiScale), Math.round(300 * Scaling.uiScale))

            anchors {
                fill: parent
                topMargin: Math.round(10 * Scaling.uiScale)
                bottomMargin: Math.round(10 * Scaling.uiScale)
                leftMargin: Math.round(8 * Scaling.uiScale)
                rightMargin: Math.round(8 * Scaling.uiScale)
            }
            spacing: Math.round(10 * Scaling.uiScale)

            ConfigurationEditor {
                id: publishersConfigEditor
                Layout.fillWidth: true
                visible: !!(root.workspace && root.workspace.publishersConfigAdapter)
                configAdapter: root.workspace ? root.workspace.publishersConfigAdapter : null
                workspace: root.workspace
                flattenFields: false
            }
        }
    }
}
