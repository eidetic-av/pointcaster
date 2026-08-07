import QtQml.Models
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.kdab.dockwidgets as KDDW

import Pointcaster 1.0

KDDW.DockWidget {
    id: root
    uniqueName: "sessionPropertiesWindow"
    title: "Session"

    property var workspace: null

    readonly property var sessionAdapter: workspace ? workspace.selectedSessionAdapter : null
    readonly property var operators: workspace ? workspace.selectedSessionOperatorAdapters : []
    readonly property string sessionId: sessionAdapter ? String(sessionAdapter.id) : ""

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

            ScrollView {
                id: scrollView
                Layout.fillWidth: true
                Layout.fillHeight: true

                Column {
                    width: scrollView.availableWidth
                    spacing: sessionConfigEditor.groupSpacing

                    ConfigurationEditor {
                        id: sessionConfigEditor
                        configAdapter: root.sessionAdapter
                        workspace: root.workspace
                        flattenFields: false
                        width: parent.width
                    }

                    OperatorPipelineEditor {
                        workspace: root.workspace
                        operators: root.operators
                        pipelineAdapter: root.operator_pipelineAdapter
                        width: parent.width

                        onAddOperatorRequested: operatorType => root.workspace.addOperatorToSession(root.sessionId, operatorType)
                        onRemoveOperatorRequested: operatorIndex => root.workspace.removeOperatorFromSession(root.sessionId, operatorIndex)
                    }
                }
            }

            // SessionTimeline {
            //     adapter: root.workspace.sessionPointCloudAdapterFor(sessionAdapter.id)
            //     Layout.fillWidth: true
            // }

            // // Session picker (only when there's more than one session)
            // ComboBox {
            //     id: sessionPicker
            //     Layout.fillWidth: true
            //     visible: root.workspace ? root.workspace.sessionAdapters.length > 1 : false

            //     model: root.workspace ? root.workspace.sessionAdapters : []
            //     textRole: ""

            //     displayText: root.sessionAdapter ? String(root.sessionAdapter.label || root.sessionAdapter.id) : ""

            //     delegate: ItemDelegate {
            //         width: sessionPicker.width
            //         text: modelData ? String(modelData.label || modelData.id) : ""
            //         font: Scaling.uiFont
            //         highlighted: sessionPicker.highlightedIndex === index
            //         onClicked: {
            //             sessionPicker.currentIndex = index;
            //             sessionPicker.popup.close();
            //             if (modelData)
            //                 root.workspace.selectedSessionId = String(modelData.id);
            //         }
            //     }
            // }

            // Text {
            //     visible: !root.sessionAdapter
            //     text: "No session selected"
            //     font: Scaling.uiFont
            //     color: ThemeColors.placeholderText
            //     Layout.fillWidth: true
            // }

        }
    }
}
