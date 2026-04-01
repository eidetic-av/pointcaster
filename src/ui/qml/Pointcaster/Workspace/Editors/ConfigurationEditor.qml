import QtQuick
import QtQuick.Dialogs
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Column {
    id: root

    required property var configAdapter
    required property var workspace

    property bool flattenFields: true

    readonly property int labelColumnWidth: WorkspaceState.labelColumnWidth
    property int minLabelColumnWidth: Math.round(40 * Scaling.uiScale)
    property int minValueColumnWidth: Math.round(170 * Scaling.uiScale)

    property int groupSpacing: Math.round(8 * Scaling.uiScale)
    property int groupInnerPaddingY: Math.round(4 * Scaling.uiScale)

    spacing: Math.round(10 * Scaling.uiScale)

    Repeater {
        model: configAdapter ? configAdapter.childPaths() : []
        delegate: configurationNode
    }

    Component {
        id: configurationNode

        Item {
            id: nodeRoot

            readonly property string defaultPath: modelData[0]

            readonly property bool nested: defaultPath.includes("/")
            readonly property string parentConfigName: root.configAdapter.parentConfigurationName(defaultPath)
            readonly property string parentKey: root.configAdapter.value("id") + "/" + parentConfigName

            property string headerText: parentConfigName
            property int headerHeight: Math.round(28 * Scaling.uiScale)
            property int fieldHeight: Math.round(28 * Scaling.uiScale)
            property bool expanded: true

            width: root.width
            height: expanded ? content.height : headerHeight

            // bindings for folding / unfolding configuration nodes
            // and serializing that in our workspace layout
            onExpandedChanged: function () {
                root.workspace.setFoldedProperty(parentKey, expanded);
            }
            function syncFoldedProperties() {
                // default to expanded if unset
                var existingValue = root.workspace.foldedPropertyPaths[nodeRoot.parentKey];
                nodeRoot.expanded = existingValue === undefined ? true : existingValue;
            }
            Connections {
                target: root.workspace
                function onFoldedPropertyPathsChanged() {
                    syncFoldedProperties();
                }
            }
            Component.onCompleted: function () {
                syncFoldedProperties();
            }

            Rectangle {
                id: nodeBackground
                anchors.fill: parent
                color: ThemeColors.dark
                border.width: nodeRoot.expanded ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0
                border.color: ThemeColors.almostdark
            }

            Rectangle {
                id: nodeHeaderBackground
                height: nodeRoot.headerHeight
                anchors {
                    top: parent.top
                    left: parent.left
                    right: parent.right
                }
                MouseArea {
                    id: mouseArea
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: nodeRoot.expanded = !nodeRoot.expanded
                }
                color: mouseArea.containsMouse ? ThemeColors.middark : (nodeRoot.expanded ? ThemeColors.almostdark : ThemeColors.dark)
                border.color: ThemeColors.almostdark
                border.width: nodeRoot.expanded ? 0 : Math.max(1, Math.round(1 * Scaling.uiScale))
            }

            Column {
                id: content

                Row {
                    id: header

                    // only show the header if unhidden fields actually exist for this configuration node
                    visible: fieldRepeater.count > 0
                    height: nodeRoot.headerHeight
                    width: parent.width
                    spacing: Math.round(5 * Scaling.uiScale)

                    Image {
                        id: headerArrowIcon

                        width: Math.round(12 * Scaling.uiScale)
                        anchors.leftMargin: header.spacing
                        anchors.verticalCenter: parent.verticalCenter

                        fillMode: Image.PreserveAspectFit
                        source: nodeRoot.expanded ? FontAwesome.icon("solid/caret-down") : FontAwesome.icon("solid/caret-right")
                        opacity: 0.75
                    }

                    Text {
                        text: nodeRoot.headerText

                        elide: Text.ElideRight
                        font: Scaling.uiFont
                        color: ThemeColors.text

                        width: parent.width - headerArrowIcon.width - header.spacing
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }

                Repeater {
                    id: fieldRepeater

                    // only create field entries if they're not marked 'hidden'
                    model: modelData.filter(function (path) {
                        return !root.configAdapter.isHidden(path);
                    })

                    delegate: RowLayout {
                        visible: nodeRoot.expanded
                        height: nodeRoot.fieldHeight

                        Text {
                            id: label
                            text: StringUtils.titleFromSnake(StringUtils.leafName(modelData))
                            color: ThemeColors.text
                            font: Scaling.uiFont

                            Layout.preferredWidth: root.labelColumnWidth
                            Layout.minimumWidth: root.minLabelColumnWidth
                            clip: true

                            Layout.leftMargin: Math.round(5 * Scaling.uiScale)

                            InfoToolTip {
                                visible: labelHover.hovered
                                textValue: "osc/address/" + modelData
                            }
                            HoverHandler {
                                id: labelHover
                            }
                        }

                        // Column width drag handle (shared global setting)
                        Rectangle {
                            id: dividerHandle
                            width: Math.round(5 * Scaling.uiScale)
                            Layout.fillHeight: true
                            color: "transparent"

                            Rectangle {
                                height: parent.height

                                width: (dividerMouseArea.containsMouse || dividerMouseArea.drag.active) ? Math.round(3 * Scaling.uiScale) : Math.max(1, Math.round(1 * Scaling.uiScale))

                                color: dividerMouseArea.drag.active ? ThemeColors.mid : (dividerMouseArea.containsMouse ? ThemeColors.middark : ThemeColors.almostdark)

                                Behavior on width {
                                    NumberAnimation {
                                        duration: Math.round(120 * Scaling.uiScale)
                                        easing.type: Easing.InCubic
                                    }
                                }

                                Behavior on color {
                                    ColorAnimation {
                                        duration: Math.round(90 * Scaling.uiScale)
                                        easing.type: Easing.InCubic
                                    }
                                }
                            }

                            MouseArea {
                                id: dividerMouseArea
                                anchors.fill: parent
                                cursorShape: Qt.SplitHCursor
                                hoverEnabled: true
                                property bool dragging: false

                                onPressed: dragging = true
                                onReleased: dragging = false
                                onCanceled: dragging = false

                                onPositionChanged: {
                                    if (!dragging)
                                        return;
                                    WorkspaceState.labelColumnWidth = Math.round(dividerHandle.x + mouseX);
                                }
                            }
                        }

                        Loader {
                            id: valueContainer

                            property string path: modelData
                            readonly property string typeName: root.configAdapter.typeName(modelData).toLowerCase()

                            Layout.fillWidth: true
                            Layout.minimumWidth: root.minValueColumnWidth

                            asynchronous: false

                            sourceComponent: {
                                if (root.configAdapter.isEnum(modelData))
                                    return enumEditor;
                                if (typeName === "int" || typeName === "int32" || typeName === "int32_t" || typeName === "integer")
                                    return intEditor;
                                if (typeName === "float" || typeName === "float32" || typeName === "float32_t" || typeName === "double" || typeName === "real" || typeName === "number")
                                    return floatEditor;
                                if (typeName === "pc::float3" || typeName === "float3")
                                    return float3Editor;
                                if (typeName === "bool")
                                    return boolEditor;
                                return stringEditor;
                            }
                        }
                    }
                }
            }
        }
    }

    //
    // FIELD EDITOR COMPONENTS
    //

    Component {
        id: stringEditor

        Row {
            width: parent.width

            TextField {
                id: valueField
                font: Scaling.uiFont

                width: parent.width - (fileOpenButton.visible ? fileOpenButton.width : 0) - Math.round(Scaling.uiScale * 5)

                background: Rectangle {
                    color: "transparent"
                    border.color: valueField.focus ? ThemeColors.highlight : "transparent"
                    border.width: Math.max(1, Math.round(1 * Scaling.uiScale))
                    radius: 0
                }

                text: root.configAdapter ? String(root.configAdapter.value(path)) : ""
                readOnly: root.configAdapter ? (root.configAdapter.isDisabled(path) || root.configAdapter.isFileOpener(path)) : false

                color: readOnly ? ThemeColors.readOnlyText : ThemeColors.text

                onEditingFinished: {
                    if (!root.configAdapter)
                        return;
                    root.configAdapter.set(path, text);
                }

                Connections {
                    target: root.configAdapter
                    function onFieldChanged(changedPath) {
                        if (String(changedPath) !== path)
                            return;
                        valueField.text = root.configAdapter ? String(root.configAdapter.value(path)) : "";
                    }
                }

                InfoToolTip {
                    textValue: valueField.text
                }
            }

            IconButton {
                id: fileOpenButton
                visible: root.configAdapter ? root.configAdapter.isFileOpener(path) : false

                iconSource: FontAwesome.icon("solid/file-import")
                iconSize: Math.round(12 * Scaling.uiScale)
                topPadding: Math.round(4 * Scaling.uiScale)
                bottomPadding: Math.round(4 * Scaling.uiScale)
                leftPadding: Math.round(5 * Scaling.uiScale)
                rightPadding: Math.round(5 * Scaling.uiScale)

                onClicked: fileOpenDialog.open()
            }

            FileDialog {
                id: fileOpenDialog
                nameFilters: [qsTr("PLY files (*.ply)"), qsTr("All files (*)")]
                onAccepted: root.configAdapter.set(path, selectedFile)
            }
        }
    }

    Component {
        id: intEditor
        DragInt {
            id: intEditorControl
            font: Scaling.uiFont

            minValue: root.configAdapter ? root.configAdapter.minMax(path)[0] : undefined
            maxValue: root.configAdapter ? root.configAdapter.minMax(path)[1] : undefined
            defaultValue: root.configAdapter ? root.configAdapter.defaultValue(path) : undefined

            enabled: root.configAdapter ? !root.configAdapter.isDisabled(path) : true

            boundValue: {
                var n = Number(root.configAdapter.value(path));
                return isNaN(n) ? 0 : Math.trunc(n);
            }

            onCommitValue: function (v) {
                root.configAdapter.set(path, v);
            }

            Connections {
                target: root.configAdapter
                function onFieldChanged(changedPath) {
                    if (String(changedPath) !== path)
                        return;
                    var n = Number(root.configAdapter.value(path));
                    intEditorControl.boundValue = isNaN(n) ? 0 : Math.trunc(n);
                }
            }
        }
    }

    Component {
        id: floatEditor
        DragFloat {
            id: floatEditorControl
            font: Scaling.uiFont

            minValue: root.configAdapter ? root.configAdapter.minMax(path)[0] : undefined
            maxValue: root.configAdapter ? root.configAdapter.minMax(path)[1] : undefined
            defaultValue: root.configAdapter ? root.configAdapter.defaultValue(path) : undefined

            enabled: root.configAdapter ? !root.configAdapter.isDisabled(path) : true

            boundValue: {
                var n = Number(root.configAdapter.value(path));
                return isNaN(n) ? 0.0 : n;
            }

            onCommitValue: function (v) {
                root.configAdapter.set(path, v);
            }

            Connections {
                target: root.configAdapter
                function onFieldChanged(changedPath) {
                    if (String(changedPath) !== path)
                        return;
                    var n = Number(root.configAdapter.value(path));
                    floatEditorControl.boundValue = isNaN(n) ? 0.0 : n;
                }
            }
        }
    }

    Component {
        id: float3Editor
        DragFloat3 {
            id: float3
            font: Scaling.uiFont
            axisFont: Scaling.uiSmallFont

            minValue: root.configAdapter ? root.configAdapter.minMax(path)[0] : undefined
            maxValue: root.configAdapter ? root.configAdapter.minMax(path)[1] : undefined
            defaultValue: root.configAdapter ? root.configAdapter.defaultValue(path) : undefined

            enabled: root.configAdapter ? !root.configAdapter.isDisabled(path) : true

            boundValue: {
                if (!root.configAdapter)
                    return Qt.vector3d(0, 0, 0);
                const v = root.configAdapter.value(path);
                return Qt.vector3d(Number(v.x) || 0, Number(v.y) || 0, Number(v.z) || 0);
            }

            onCommitValue: function (v3) {
                if (!root.configAdapter)
                    return;
                root.configAdapter.set(path, v3);
            }

            Connections {
                target: root.configAdapter
                function onFieldChanged(changedPath) {
                    if (String(changedPath) !== path)
                        return;
                    const v = root.configAdapter.value(path);
                    float3.boundValue = Qt.vector3d(Number(v.x) || 0, Number(v.y) || 0, Number(v.z) || 0);
                }
            }
        }
    }

    Component {
        id: enumEditor
        EnumSelector {
            id: enumSelector
            font: Scaling.uiFont

            enabled: root.configAdapter ? !root.configAdapter.isDisabled(path) : true
            opacity: enabled ? 1.0 : 0.66

            options: root.configAdapter.enumOptions(path)
            boundValue: {
                var n = Number(root.configAdapter.value(path));
                return isNaN(n) ? 0 : Math.trunc(n);
            }

            onCommitValue: function (v) {
                root.configAdapter.set(path, v);
            }

            Connections {
                target: root.configAdapter
                function onFieldChanged(changedPath) {
                    if (String(changedPath) !== path)
                        return;
                    var n = Number(root.configAdapter.value(path));
                    enumSelector.boundValue = isNaN(n) ? 0 : Math.trunc(n);
                }
            }
        }
    }

    Component {
        id: boolEditor

        CheckBox {
            id: boolCheckBox
            enabled: root.configAdapter ? !root.configAdapter.isDisabled(path) : true
            opacity: enabled ? 1.0 : 0.66
            checked: !!root.configAdapter.value(path)
            onCheckedChanged: function () {
                root.configAdapter.set(path, checked);
            }
            Connections {
                target: root.configAdapter
                function onFieldChanged(changedPath) {
                    if (String(changedPath) !== path)
                        return;
                    boolCheckBox.checked = !!root.configAdapter.value(path);
                }
            }
        }
    }
}
