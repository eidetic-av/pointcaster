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

        PaddedScrollView {
            id: publishersScrollView
            clip: true
            anchors {
                fill: parent
                topMargin: Math.round(10 * Scaling.uiScale)
                bottomMargin: Math.round(10 * Scaling.uiScale)
                leftMargin: Math.round(8 * Scaling.uiScale)
                rightMargin: Math.round(8 * Scaling.uiScale)
            }

            ConfigurationEditor {
                id: publishersConfigEditor
                width: publishersScrollView.availableWidth
                visible: !!(root.workspace && root.workspace.publishersConfigAdapter)
                configAdapter: root.workspace ? root.workspace.publishersConfigAdapter : null
                workspace: root.workspace
                flattenFields: false
            }
        }
    }
}
