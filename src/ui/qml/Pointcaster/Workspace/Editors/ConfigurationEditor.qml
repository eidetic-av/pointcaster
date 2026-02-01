import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    property var model: null
    property string emptyTitle: "Properties"

    property bool flattenFields: true

    // ---- Layout
    readonly property int labelColumnWidth: WorkspaceState.labelColumnWidth
    property int minLabelColumnWidth: Math.round(40 * Scaling.uiScale)
    property int minValueColumnWidth: Math.round(170 * Scaling.uiScale)
    property int outerHorizontalMargin: Math.round(8 * Scaling.uiScale)
    property int outerTopMargin: Math.round(4 * Scaling.uiScale)

    // Spacing/padding tuning
    property int groupSpacing: Math.round(8 * Scaling.uiScale)
    property int groupInnerPaddingY: Math.round(4 * Scaling.uiScale)

    readonly property real _viewportWidth: {
        if (ScrollView.view)
            return ScrollView.view.width;
        if (parent)
            return parent.width;
        return 0;
    }
    width: _viewportWidth

    function _paths() {
        if (!root.model || !root.model.fieldPaths)
            return [];
        const p = root.model.fieldPaths();
        return p ? p : [];
    }

    function _leafName(path) {
        if (!path || path.length === 0)
            return "";
        const parts = String(path).split("/");
        return parts.length > 0 ? parts[parts.length - 1] : String(path);
    }

    // ---- Build group tree from fieldPaths() ----
    // node: { title, prefix, leafNames: [], children: [] }
    function _buildGroupTree(allPaths) {
        const rootNode = {
            title: root.model ? (root.model.displayName() + " Properties") : root.emptyTitle,
            prefix: "",
            leafNames: [],
            children: []
        };

        function ensureChild(parentNode, seg, accumulatedPrefix) {
            for (let i = 0; i < parentNode.children.length; ++i) {
                if (parentNode.children[i].prefix === accumulatedPrefix)
                    return parentNode.children[i];
            }
            const child = {
                title: StringUtils.titleFromSnake(seg),
                prefix: accumulatedPrefix,
                leafNames: [],
                children: []
            };
            parentNode.children.push(child);
            return child;
        }

        for (let i = 0; i < allPaths.length; ++i) {
            const path = String(allPaths[i]);
            if (!path)
                continue;

            const parts = path.split("/");
            if (parts.length === 1) {
                rootNode.leafNames.push(parts[0]);
                continue;
            }

            let node = rootNode;
            let prefixSoFar = "";
            for (let p = 0; p < parts.length; ++p) {
                const seg = parts[p];
                const isLeaf = (p === parts.length - 1);

                if (isLeaf) {
                    node.leafNames.push(seg);
                    break;
                }

                prefixSoFar = prefixSoFar.length ? (prefixSoFar + "/" + seg) : seg;
                node = ensureChild(node, seg, prefixSoFar);
            }
        }

        function sortNode(node) {
            node.leafNames.sort();
            node.children.sort((a, b) => a.title.localeCompare(b.title));
            for (let i = 0; i < node.children.length; ++i)
                sortNode(node.children[i]);
        }
        sortNode(rootNode);
        return rootNode;
    }

    function _flattenedGroup(allPaths) {
        return [
            {
                title: root.model ? (root.model.displayName() + " Properties") : root.emptyTitle,
                prefix: "",
                // full paths in leafNames
                leafNames: allPaths.map(p => String(p)),
                children: []
            }
        ];
    }

    // Flatten a tree into a list of nodes (so groups are siblings)
    function _flattenTreeToGroups(treeRoot) {
        const out = [];

        function visit(node) {
            out.push({
                title: node.title,
                prefix: node.prefix,
                leafNames: node.leafNames,
                children: [] // IMPORTANT: not used in this rendering mode
            });
            for (let i = 0; i < node.children.length; ++i)
                visit(node.children[i]);
        }

        visit(treeRoot);
        return out;
    }

    function _computeGroups() {
        if (!root.model)
            return [];

        const allPaths = root._paths();

        if (root.flattenFields)
            return root._flattenedGroup(allPaths);

        const tree = root._buildGroupTree(allPaths);
        return root._flattenTreeToGroups(tree);
    }

    // ---------- InspectorGroup Component ----------
    Component {
        id: inspectorGroupComponent

        Item {
            id: groupRoot

            property var model: null
            property string title: ""
            property string prefix: ""
            property var leafNames: []

            property bool expanded: true

            // Bind to global/shared sizing so every instance stays synced
            readonly property int labelColumnWidth: WorkspaceState.labelColumnWidth
            readonly property int minLabelColumnWidth: root.minLabelColumnWidth
            readonly property int minValueColumnWidth: root.minValueColumnWidth

            readonly property var groupPaths: {
                if (!leafNames)
                    return [];
                const out = [];
                for (let i = 0; i < leafNames.length; ++i) {
                    const leaf = String(leafNames[i]);
                    if (!leaf)
                        continue;

                    // flattenFields passes full paths in leafNames
                    if (leaf.indexOf("/") >= 0) {
                        out.push(leaf);
                    } else {
                        out.push(prefix && prefix.length ? (prefix + "/" + leaf) : leaf);
                    }
                }
                return out;
            }

            implicitHeight: header.height + (expanded ? body.implicitHeight : 0)

            Rectangle {
                id: inspectorGroupRect
                anchors.left: parent.left
                anchors.right: parent.right
                height: groupRoot.implicitHeight

                color: ThemeColors.dark
                border.color: ThemeColors.mid
                border.width: Math.max(1, Math.round(1 * Scaling.uiScale))
                clip: true

                Rectangle {
                    id: header
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    height: Math.round(28 * Scaling.uiScale)
                    color: headerMouseArea.containsMouse ? ThemeColors.mid : (groupRoot.expanded ? ThemeColors.middark : ThemeColors.dark)
                    border.color: groupRoot.expanded ? ThemeColors.mid : ThemeColors.middark
                    border.width: Math.max(1, Math.round(1 * Scaling.uiScale))

                    Image {
                        id: headerArrowIcon
                        width: Math.round(12 * Scaling.uiScale)
                        height: Math.round(12 * Scaling.uiScale)
                        fillMode: Image.PreserveAspectFit
                        source: groupRoot.expanded ? FontAwesome.icon("solid/caret-down") : FontAwesome.icon("solid/caret-right")
                        opacity: 0.75
                        anchors.left: parent.left
                        anchors.leftMargin: Math.round(3 * Scaling.uiScale)
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    Text {
                        id: headerText
                        anchors.left: headerArrowIcon.left
                        anchors.right: parent.right
                        anchors.leftMargin: Math.round(15 * Scaling.uiScale)
                        anchors.rightMargin: Math.round(10 * Scaling.uiScale)
                        anchors.verticalCenter: parent.verticalCenter
                        text: groupRoot.title
                        elide: Text.ElideRight
                        font: Scaling.uiFont
                        color: ThemeColors.text
                    }

                    MouseArea {
                        id: headerMouseArea
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: groupRoot.expanded = !groupRoot.expanded
                    }
                }

                Item {
                    id: body
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: header.bottom
                    visible: groupRoot.expanded

                    implicitHeight: (fieldsArea.visible ? fieldsContainer.implicitHeight : 0) + root.groupInnerPaddingY

                    Item {
                        id: fieldsArea
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.topMargin: root.groupInnerPaddingY
                        visible: groupRoot.model && groupRoot.groupPaths.length > 0
                        height: fieldsContainer.implicitHeight

                        readonly property int _labelLeftMargin: Math.round(6 * Scaling.uiScale)
                        readonly property int _valueLeftOverlap: Math.round(5 * Scaling.uiScale)
                        readonly property real availableInnerWidth: Math.max(0, width - _labelLeftMargin + _valueLeftOverlap)

                        readonly property int effectiveLabelWidth: {
                            const preferred = groupRoot.labelColumnWidth;

                            let maxLabel = Math.floor(availableInnerWidth - groupRoot.minValueColumnWidth);
                            if (!Number.isFinite(maxLabel))
                                maxLabel = 0;

                            let w = Math.max(groupRoot.minLabelColumnWidth, preferred);
                            w = Math.min(w, maxLabel);
                            return Math.max(0, w);
                        }

                        readonly property int dragMinLabel: groupRoot.minLabelColumnWidth
                        readonly property int dragMaxLabel: Math.max(groupRoot.minLabelColumnWidth, Math.floor(availableInnerWidth - groupRoot.minValueColumnWidth))

                        Column {
                            id: fieldsContainer
                            width: fieldsArea.width
                            spacing: 0

                            Repeater {
                                model: groupRoot.groupPaths.length

                                delegate: Item {
                                    required property int index
                                    readonly property string path: String(groupRoot.groupPaths[index])

                                    width: fieldsContainer.width
                                    height: Math.max(nameText.implicitHeight, valueContainer.implicitHeight) + Math.round(6 * Scaling.uiScale)
                                    visible: groupRoot.model && path.length > 0 && !groupRoot.model.isHidden(path)

                                    Item {
                                        id: labelContainer
                                        anchors.top: parent.top
                                        anchors.bottom: parent.bottom
                                        anchors.left: parent.left
                                        anchors.leftMargin: fieldsArea._labelLeftMargin
                                        width: fieldsArea.effectiveLabelWidth

                                        Text {
                                            id: nameText
                                            anchors.left: parent.left
                                            anchors.right: parent.right
                                            anchors.rightMargin: Math.round(10 * Scaling.uiScale)
                                            anchors.verticalCenter: parent.verticalCenter
                                            horizontalAlignment: Text.AlignLeft
                                            color: ThemeColors.text
                                            text: groupRoot.model ? StringUtils.titleFromSnake(root._leafName(path)) : ""
                                            font: Scaling.uiFont
                                            elide: Text.ElideRight
                                        }

                                        InfoToolTip {
                                            visible: labelHover.hovered
                                            textValue: groupRoot.model ? ("osc/address/" + path) : ""
                                            x: Math.max((labelContainer.width - implicitWidth) * 0.5, Math.round(4 * Scaling.uiScale))
                                            y: labelContainer.height + Math.round(4 * Scaling.uiScale)
                                        }

                                        HoverHandler {
                                            id: labelHover
                                        }
                                    }

                                    Loader {
                                        id: valueContainer
                                        anchors.left: labelContainer.right
                                        anchors.leftMargin: -fieldsArea._valueLeftOverlap
                                        anchors.right: parent.right
                                        anchors.verticalCenter: parent.verticalCenter
                                        z: 3

                                        readonly property string typeName: groupRoot.model ? String(groupRoot.model.typeName(path)).toLowerCase() : ""

                                        sourceComponent: {
                                            if (groupRoot.model && groupRoot.model.isEnum(path))
                                                return enumEditor;
                                            if (typeName === "int" || typeName === "int32" || typeName === "int32_t" || typeName === "integer")
                                                return intEditor;
                                            if (typeName === "float" || typeName === "float32" || typeName === "float32_t" || typeName === "double" || typeName === "real" || typeName === "number")
                                                return floatEditor;
                                            if (typeName === "pc::float3" || typeName === "float3")
                                                return float3Editor;
                                            return stringEditor;
                                        }
                                    }

                                    Component {
                                        id: stringEditor
                                        TextField {
                                            id: valueField
                                            font: Scaling.uiFont

                                            background: Rectangle {
                                                color: "transparent"
                                                border.color: valueField.focus ? ThemeColors.highlight : "transparent"
                                                border.width: Math.max(1, Math.round(1 * Scaling.uiScale))
                                                radius: 0
                                            }

                                            text: groupRoot.model ? String(groupRoot.model.value(path)) : ""
                                            enabled: groupRoot.model ? !groupRoot.model.isDisabled(path) : false
                                            opacity: enabled ? 1.0 : 0.66

                                            onEditingFinished: {
                                                if (!groupRoot.model)
                                                    return;
                                                groupRoot.model.set(path, text);
                                            }

                                            Connections {
                                                target: groupRoot.model
                                                function onFieldChanged(changedPath) {
                                                    if (String(changedPath) !== path)
                                                        return;
                                                    valueField.text = groupRoot.model ? String(groupRoot.model.value(path)) : "";
                                                }
                                            }
                                        }
                                    }

                                    Component {
                                        id: intEditor
                                        DragInt {
                                            id: intEditorControl
                                            font: Scaling.uiFont

                                            minValue: groupRoot.model ? groupRoot.model.minMax(path)[0] : undefined
                                            maxValue: groupRoot.model ? groupRoot.model.minMax(path)[1] : undefined
                                            defaultValue: groupRoot.model ? groupRoot.model.defaultValue(path) : undefined

                                            boundValue: {
                                                if (!groupRoot.model)
                                                    return 0;
                                                var n = Number(groupRoot.model.value(path));
                                                return isNaN(n) ? 0 : Math.trunc(n);
                                            }

                                            onCommitValue: function (v) {
                                                if (!groupRoot.model)
                                                    return;
                                                groupRoot.model.set(path, v);
                                            }

                                            Connections {
                                                target: groupRoot.model
                                                function onFieldChanged(changedPath) {
                                                    if (String(changedPath) !== path)
                                                        return;
                                                    var n = Number(groupRoot.model.value(path));
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

                                            minValue: groupRoot.model ? groupRoot.model.minMax(path)[0] : undefined
                                            maxValue: groupRoot.model ? groupRoot.model.minMax(path)[1] : undefined
                                            defaultValue: groupRoot.model ? groupRoot.model.defaultValue(path) : undefined

                                            boundValue: {
                                                if (!groupRoot.model)
                                                    return 0.0;
                                                var n = Number(groupRoot.model.value(path));
                                                return isNaN(n) ? 0.0 : n;
                                            }

                                            onCommitValue: function (v) {
                                                if (!groupRoot.model)
                                                    return;
                                                groupRoot.model.set(path, v);
                                            }

                                            Connections {
                                                target: groupRoot.model
                                                function onFieldChanged(changedPath) {
                                                    if (String(changedPath) !== path)
                                                        return;
                                                    var n = Number(groupRoot.model.value(path));
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

                                            property var minMax: groupRoot.model ? groupRoot.model.minMax(path) : undefined
                                            minValue: minMax ? minMax[0] : undefined
                                            maxValue: minMax ? minMax[1] : undefined

                                            boundValue: {
                                                if (!groupRoot.model)
                                                    return Qt.vector3d(0, 0, 0);
                                                const v = groupRoot.model.value(path);
                                                return Qt.vector3d(Number(v.x) || 0, Number(v.y) || 0, Number(v.z) || 0);
                                            }

                                            onCommitValue: function (v3) {
                                                if (!groupRoot.model)
                                                    return;
                                                groupRoot.model.set(path, v3);
                                            }

                                            Connections {
                                                target: groupRoot.model
                                                function onFieldChanged(changedPath) {
                                                    if (String(changedPath) !== path)
                                                        return;
                                                    const v = groupRoot.model.value(path);
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

                                            enabled: groupRoot.model ? !groupRoot.model.isDisabled(path) : false
                                            opacity: enabled ? 1.0 : 0.66

                                            options: groupRoot.model ? groupRoot.model.enumOptions(path) : []
                                            boundValue: {
                                                if (!groupRoot.model)
                                                    return 0;
                                                var n = Number(groupRoot.model.value(path));
                                                return isNaN(n) ? 0 : Math.trunc(n);
                                            }

                                            onCommitValue: function (v) {
                                                if (!groupRoot.model)
                                                    return;
                                                groupRoot.model.set(path, v);
                                            }

                                            Connections {
                                                target: groupRoot.model
                                                function onFieldChanged(changedPath) {
                                                    if (String(changedPath) !== path)
                                                        return;
                                                    var n = Number(groupRoot.model.value(path));
                                                    enumSelector.boundValue = isNaN(n) ? 0 : Math.trunc(n);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Column width drag handle (shared global setting)
                        Rectangle {
                            id: dividerHandle
                            x: (fieldsArea.effectiveLabelWidth - width / 2) + Math.max(1, Math.round(1 * Scaling.uiScale))
                            z: 99
                            width: Math.round(5 * Scaling.uiScale)
                            anchors.top: parent.top
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: Math.max(1, Math.round(1 * Scaling.uiScale))
                            color: "transparent"

                            Rectangle {
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.horizontalCenter: parent.horizontalCenter
                                height: parent.height
                                width: (dividerMouseArea.containsMouse || dividerMouseArea.drag.active) ? Math.round(3 * Scaling.uiScale) : Math.max(1, Math.round(1 * Scaling.uiScale))

                                Behavior on width {
                                    NumberAnimation {
                                        duration: Math.round(120 * Scaling.uiScale)
                                        easing.type: Easing.InCubic
                                    }
                                }

                                color: dividerMouseArea.drag.active ? ThemeColors.midlight : (dividerMouseArea.containsMouse ? ThemeColors.mid : ThemeColors.middark)

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

                                    const dividerCentreX = dividerHandle.x + mouseX;
                                    const desiredLabelWidth = dividerCentreX - fieldsArea._labelLeftMargin;
                                    const clamped = Math.max(fieldsArea.dragMinLabel, Math.min(fieldsArea.dragMaxLabel, desiredLabelWidth));

                                    // single source of truth
                                    WorkspaceState.labelColumnWidth = Math.round(clamped);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    implicitHeight: outerTopMargin + (root.model ? groupsColumn.implicitHeight : noSelectionText.implicitHeight + Math.round(8 * Scaling.uiScale))

    Text {
        id: noSelectionText
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.leftMargin: Math.round(10 * Scaling.uiScale)
        anchors.rightMargin: Math.round(10 * Scaling.uiScale)
        anchors.topMargin: Math.round(3 * Scaling.uiScale)
        visible: !root.model
        text: "No device selected"
        font: Scaling.uiFont
        color: ThemeColors.text
    }

    Column {
        id: groupsColumn
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.leftMargin: root.outerHorizontalMargin
        anchors.rightMargin: root.outerHorizontalMargin
        anchors.topMargin: root.outerTopMargin

        spacing: root.groupSpacing
        visible: !!root.model

        Instantiator {
            id: topInstantiator
            model: root.model ? root._computeGroups().length : 0
            delegate: inspectorGroupComponent

            onObjectAdded: function (index, object) {
                const node = root._computeGroups()[index];

                object.parent = groupsColumn;

                object.model = root.model;
                object.title = node.title;
                object.prefix = node.prefix;
                object.leafNames = node.leafNames;

                object.expanded = true;
                object.anchors.left = groupsColumn.left;
                object.anchors.right = groupsColumn.right;
            }

            onObjectRemoved: function (index, object) {
                if (object)
                    object.destroy();
            }
        }
    }
}
