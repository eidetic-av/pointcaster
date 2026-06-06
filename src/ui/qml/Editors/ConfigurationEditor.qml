import QtQuick
import QtQuick.Dialogs
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Column {
    id: root

    required property var configAdapter
    required property var workspace

    property bool flattenFields: true

    readonly property int labelColumnWidth: Workspace.labelColumnWidth
    property int minLabelColumnWidth: Math.round(40 * Scaling.uiScale)
    property int minValueColumnWidth: Math.round(170 * Scaling.uiScale)

    property int groupSpacing: Math.round(8 * Scaling.uiScale)
    property int groupInnerPaddingY: Math.round(4 * Scaling.uiScale)

    spacing: groupSpacing

    Repeater {
        model: configAdapter ? configAdapter.childPaths() : []
        delegate: configurationNode
    }

    Component {
        id: configurationNode

        Rectangle {
            id: nodeRoot

            readonly property string defaultPath: modelData[0]

            readonly property bool nested: defaultPath.includes("/")
            readonly property string parentConfigName: root.configAdapter.parentConfigurationName(defaultPath)
            readonly property string parentKey: root.configAdapter.value("id") + "/" + parentConfigName

            readonly property bool flattened: root.flattenFields
            readonly property bool fieldsVisible: flattened || expanded

            property string headerText: parentConfigName
            property int headerHeight: Math.round(28 * Scaling.uiScale)
            property int fieldHeight: Math.round(28 * Scaling.uiScale)
            property bool expanded: true

            width: root.width
            implicitHeight: contentColumn.implicitHeight

            color: flattened ? "transparent" : ThemeColors.dark
            border.width: (!flattened && expanded) ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0
            border.color: ThemeColors.almostdark

            onExpandedChanged: {
                if (!flattened)
                    root.workspace.setFoldedProperty(parentKey, expanded);
            }

            function syncFoldedProperties() {
                if (flattened) {
                    expanded = true;
                    return;
                }
                var existingValue = root.workspace.foldedPropertyPaths[parentKey];
                expanded = existingValue === undefined ? true : existingValue;
            }

            Connections {
                target: root.workspace
                function onFoldedPropertyPathsChanged() {
                    nodeRoot.syncFoldedProperties();
                }
            }

            Component.onCompleted: syncFoldedProperties()

            Column {
                id: contentColumn
                width: parent.width

                // Header
                Rectangle {
                    id: nodeHeader
                    width: parent.width
                    height: nodeRoot.headerHeight
                    visible: fieldRepeater.count > 0

                    color: nodeRoot.flattened ? "transparent" : (headerMouseArea.containsMouse ? ThemeColors.middark : (nodeRoot.expanded ? ThemeColors.almostdark : ThemeColors.dark))

                    border.color: ThemeColors.almostdark
                    border.width: (!nodeRoot.flattened && !nodeRoot.expanded) ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0

                    MouseArea {
                        id: headerMouseArea
                        anchors.fill: parent
                        enabled: !nodeRoot.flattened
                        hoverEnabled: enabled
                        onClicked: nodeRoot.expanded = !nodeRoot.expanded
                    }

                    Rectangle {
                        visible: nodeRoot.flattened
                        height: Math.max(1, Math.round(1 * Scaling.uiScale))
                        anchors {
                            left: parent.left
                            right: parent.right
                            bottom: parent.bottom
                        }
                        color: ThemeColors.almostdark
                    }

                    Row {
                        anchors.fill: parent
                        spacing: Math.round(5 * Scaling.uiScale)

                        Item {
                            width: 1
                            height: 1
                        }

                        Image {
                            id: headerArrowIcon
                            visible: !nodeRoot.flattened
                            width: visible ? Math.round(12 * Scaling.uiScale) : 0
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
                            width: parent.width - Math.round(10 * Scaling.uiScale) - (headerArrowIcon.visible ? headerArrowIcon.width + parent.spacing : 0)
                            anchors.verticalCenter: parent.verticalCenter
                        }
                    }
                }

                // Fields
                Repeater {
                    id: fieldRepeater

                    model: modelData.filter(function (path) {
                        return !root.configAdapter.isHidden(path);
                    })

                    delegate: RowLayout {
                        visible: nodeRoot.fieldsVisible
                        height: visible ? nodeRoot.fieldHeight : 0

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
                                    Workspace.labelColumnWidth = Math.round(dividerHandle.x + mouseX);
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
                                if (typeName.includes("pc::float3") || typeName.includes("float3"))
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

        RowLayout {
            width: parent.width
            spacing: Math.round(Scaling.uiScale * 5)

            TextField {
                id: valueField
                font: Scaling.uiFont

                Layout.fillWidth: true

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
                        // root.configAdapter["restart"]();
                    }
                }

                InfoToolTip {
                    textValue: valueField.text
                }
            }

            IconButton {
                id: fileOpenButton
                visible: root.configAdapter ? root.configAdapter.isFileOpener(path) : false

                tooltip: "Open file..."

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

            IconButton {
                id: folderOpenButton
                visible: root.configAdapter ? root.configAdapter.isFolderOpener(path) : false

                tooltip: "Open folder..."

                iconSource: FontAwesome.icon("solid/folder-open")
                iconSize: Math.round(12 * Scaling.uiScale)
                topPadding: Math.round(4 * Scaling.uiScale)
                bottomPadding: Math.round(4 * Scaling.uiScale)
                leftPadding: Math.round(5 * Scaling.uiScale)
                rightPadding: Math.round(5 * Scaling.uiScale)

                onClicked: folderOpenDialog.open()
            }

            FolderDialog {
                id: folderOpenDialog
                onAccepted: root.configAdapter.set(path, selectedFolder)
            }

            IconButton {
                id: reloadButton
                visible: root.configAdapter ? root.configAdapter.isFileOpener(path) || root.configAdapter.isFolderOpener(path) : false

                tooltip: "Reload"

                iconSource: FontAwesome.icon("solid/arrows-rotate")
                iconSize: Math.round(12 * Scaling.uiScale)
                topPadding: Math.round(4 * Scaling.uiScale)
                bottomPadding: Math.round(4 * Scaling.uiScale)
                leftPadding: Math.round(5 * Scaling.uiScale)
                rightPadding: Math.round(5 * Scaling.uiScale)

                // onClicked: fileOpenDialog.open()

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
                var n = root.configAdapter ? Number(root.configAdapter.value(path)) : 0;
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

            options: root.configAdapter ? root.configAdapter.enumOptions(path) : undefined
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
            checked: root.configAdapter ? !!root.configAdapter.value(path) : false
            onCheckedChanged: function () {
                root.configAdapter.set(path, checked);
                // if its a button, we only want it to be momentarily checked
                if (root.configAdapter.isButton(path)) {
                    boolCheckBox.checked = false;
                }
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
