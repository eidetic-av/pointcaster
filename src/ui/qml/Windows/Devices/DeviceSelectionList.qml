import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Item {
    id: root

    property var workspace: null
    property var currentItem: list.currentItem
    property var selectedDevice: null

    property int indentStep: Math.round(14 * Scaling.uiScale)

    readonly property real deviceRowHeight: Math.max(Math.round(30 * Scaling.uiScale), Math.ceil(Scaling.pointSize * 2.1))
    readonly property real groupRowHeight: root.deviceRowHeight

    // drag state, shared across delegates
    property int dragOverIndex: -1
    property string dragOverZone: "" // "before" | "after" | "into"
    property bool dropToRoot: false

    width: parent.width

    signal activated(int index)

    function selectDevice(deviceIndex) {
        if (deviceIndex < 0)
            return;
        const rows = workspace ? workspace.deviceTreeRows : [];
        for (let i = 0; i < rows.length; ++i) {
            if (rows[i].kind === "device" && rows[i].deviceIndex === deviceIndex) {
                setSelectedIndex(i, deviceIndex);
                return;
            }
        }
    }

    function setSelectedIndex(rowIndex, deviceIndex) {
        list.currentIndex = rowIndex;
        if (deviceIndex < 0)
            return;
        workspace.selectedDeviceIndex = deviceIndex;
        selectedDevice = workspace.deviceAdapters[deviceIndex];
        activated(deviceIndex);
    }

    // 0 off, 1 mixed, 2 on -> Qt.CheckState
    function toCheckState(state) {
        if (state === 2)
            return Qt.Checked;
        if (state === 1)
            return Qt.PartiallyChecked;
        return Qt.Unchecked;
    }

    // id of the next row sharing the same parent and kind, the insert-before
    // anchor for reordering within a parent
    function nextSiblingId(rowIndex, parentId) {
        const rows = workspace ? workspace.deviceTreeRows : [];
        for (let i = rowIndex + 1; i < rows.length; ++i) {
            if (rows[i].parentId === parentId)
                return String(rows[i].id);
        }
        return "";
    }

    // drop zone from the cursor's vertical position within a row
    function zoneFor(rowKind, localY, h) {
        if (rowKind === "group") {
            if (localY < h * 0.28)
                return "before";
            if (localY > h * 0.72)
                return "after";
            return "into";
        }
        return localY < h * 0.5 ? "before" : "after";
    }

    // resolves a scene point to a row index and zone via the list geometry
    function rowAndZoneAt(sceneX, sceneY) {
        if (!workspace)
            return {
                index: -1,
                inList: false,
                zone: "",
                row: null
            };
        const vp = list.mapFromItem(null, sceneX, sceneY);
        const inList = vp.x >= 0 && vp.x <= list.width && vp.y >= 0 && vp.y <= list.height;
        if (!inList)
            return {
                index: -1,
                inList: false,
                zone: "",
                row: null
            };
        const cy = list.contentY + vp.y;
        const idx = list.indexAt(list.width / 2, cy);
        if (idx < 0)
            return {
                index: -1,
                inList: true,
                zone: "",
                row: null
            };
        const rows = workspace.deviceTreeRows;
        const row = rows[idx];
        const item = list.itemAtIndex(idx);
        const localY = item ? (cy - item.y) : 0;
        const h = item ? item.height : 1;
        return {
            index: idx,
            inList: true,
            zone: zoneFor(row.kind, localY, h),
            row: row
        };
    }

    // called continuously while a row is being dragged
    function updateDragTarget(sceneX, sceneY, sourceId) {
        const info = rowAndZoneAt(sceneX, sceneY);
        if (info.index >= 0 && !(info.row && String(info.row.id) === sourceId)) {
            dragOverIndex = info.index;
            dragOverZone = info.zone;
            dropToRoot = false;
        } else {
            dragOverIndex = -1;
            dropToRoot = info.inList; // empty space inside the list means root
        }
    }

    // called once on release
    function finishDrag(sourceId, sourceKind) {
        const idx = dragOverIndex;
        const zone = dragOverZone;
        const toRoot = dropToRoot;
        dragOverIndex = -1;
        dropToRoot = false;
        if (idx >= 0) {
            const row = workspace.deviceTreeRows[idx];
            if (row)
                performDrop(sourceId, sourceKind, row, idx, zone);
        } else if (toRoot) {
            workspace.moveDeviceNode(sourceId, "", "");
        }
    }

    function performDrop(sourceId, sourceKind, targetRow, targetRowIndex, zone) {
        if (!sourceId || sourceId.length === 0)
            return;
        if (sourceId === String(targetRow.id))
            return;

        if (zone === "into") {
            workspace.moveDeviceNode(sourceId, String(targetRow.id), "");
            return;
        }

        const parent = String(targetRow.parentId);
        const before = (zone === "before") ? String(targetRow.id) : nextSiblingId(targetRowIndex, parent);
        if (before === sourceId)
            return;
        workspace.moveDeviceNode(sourceId, parent, before);
    }

    // context menu for group operations, target set before popup
    Menu {
        id: contextMenu
        property string targetGroupId: ""
        property bool onGroup: false

        MenuItem {
            text: contextMenu.onGroup ? "New nested group" : "New group"
            onTriggered: root.workspace.createDeviceGroup("Group", contextMenu.targetGroupId)
        }
        MenuItem {
            text: "Delete group (keep devices)"
            visible: contextMenu.onGroup
            height: visible ? implicitHeight : 0
            onTriggered: root.workspace.deleteDeviceGroup(contextMenu.targetGroupId)
        }
    }

    //
    // ROW LOADER
    //

    Component {
        id: rowLoader

        Loader {
            required property var modelData
            required property int index

            width: list.width
            height: modelData.kind === "group" ? root.groupRowHeight : root.deviceRowHeight
            sourceComponent: modelData.kind === "group" ? groupDelegate : deviceDelegate
            onLoaded: {
                item.row = modelData;
                item.rowIndex = index;
            }
        }
    }

    //
    // GROUP DELEGATE
    //

    Component {
        id: groupDelegate

        Item {
            id: groupArea

            property var row: ({})
            property int rowIndex: -1
            property string nodeId: row.id !== undefined ? String(row.id) : ""
            property string nodeKind: "group"
            property real indent: (row.depth || 0) * root.indentStep
            readonly property real rowHeight: root.groupRowHeight

            width: list.width
            height: rowHeight
            z: groupDrag.active ? 1 : 0

            Rectangle {
                id: content
                width: list.width
                height: groupArea.rowHeight

                readonly property bool intoTarget: root.dragOverIndex === groupArea.rowIndex && root.dragOverZone === "into"

                color: intoTarget ? ThemeColors.mid : ThemeColors.dark
                border.color: intoTarget ? ThemeColors.highlight : ThemeColors.almostdark
                border.width: intoTarget ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0

                states: State {
                    when: groupDrag.active
                    ParentChange {
                        target: content
                        parent: dragLayer
                        x: 0
                    }
                }

                DragHandler {
                    id: groupDrag
                    target: null
                    enabled: !groupLabelBox.editing
                    xAxis.enabled: false
                    yAxis.enabled: true
                    onActiveChanged: {
                        if (active)
                            content.y = groupArea.mapToItem(dragLayer, 0, 0).y;
                        else
                            root.finishDrag(groupArea.nodeId, groupArea.nodeKind);
                    }
                    onCentroidChanged: {
                        if (active) {
                            content.y = groupArea.mapToItem(dragLayer, 0, 0).y + activeTranslation.y;
                            root.updateDragTarget(centroid.scenePosition.x, centroid.scenePosition.y, groupArea.nodeId);
                        }
                    }
                }

                TapHandler {
                    id: groupTap
                    enabled: !groupLabelBox.editing
                    acceptedButtons: Qt.LeftButton | Qt.RightButton
                    onTapped: (point, button) => {
                        list.forceActiveFocus();
                        if (button === Qt.RightButton) {
                            contextMenu.targetGroupId = groupArea.nodeId;
                            contextMenu.onGroup = true;
                            contextMenu.popup();
                        }
                    }
                    onDoubleTapped: (point, button) => {
                        if (button !== Qt.LeftButton)
                            return;
                        const p = groupLabelBox.mapFromItem(content, point.position.x, point.position.y);
                        if (p.x >= 0 && p.x <= groupLabelBox.width && p.y >= 0 && p.y <= groupLabelBox.height) {
                            groupLabelBox.editing = true;
                            groupLabelBox.startText = groupLabelInput.text;
                            groupLabelInput.forceActiveFocus();
                            groupLabelInput.selectAll();
                        }
                    }
                }

                // insert-before / insert-after indicators
                Rectangle {
                    anchors {
                        left: parent.left
                        right: parent.right
                        top: parent.top
                    }
                    height: Math.max(2, Math.round(2 * Scaling.uiScale))
                    color: ThemeColors.highlight
                    visible: root.dragOverIndex === groupArea.rowIndex && root.dragOverZone === "before"
                }
                Rectangle {
                    anchors {
                        left: parent.left
                        right: parent.right
                        bottom: parent.bottom
                    }
                    height: Math.max(2, Math.round(2 * Scaling.uiScale))
                    color: ThemeColors.highlight
                    visible: root.dragOverIndex === groupArea.rowIndex && root.dragOverZone === "after"
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: Math.round(8 * Scaling.uiScale) + groupArea.indent
                    anchors.rightMargin: Math.round(8 * Scaling.uiScale)
                    spacing: Math.round(8 * Scaling.uiScale)

                    Image {
                        id: chevron
                        Layout.alignment: Qt.AlignVCenter
                        Layout.preferredWidth: Math.round(11 * Scaling.uiScale)
                        Layout.preferredHeight: Math.round(11 * Scaling.uiScale)
                        fillMode: Image.PreserveAspectFit
                        sourceSize: Qt.size(Math.round(11 * Scaling.uiScale), Math.round(11 * Scaling.uiScale))
                        source: groupArea.row.collapsed ? FontAwesome.icon("solid/caret-right") : FontAwesome.icon("solid/caret-down")
                        opacity: 0.6

                        TapHandler {
                            onTapped: {
                                list.forceActiveFocus();
                                root.workspace.setDeviceGroupCollapsed(groupArea.nodeId, !groupArea.row.collapsed);
                            }
                        }
                    }

                    CheckBox {
                        id: groupActiveToggle
                        Layout.maximumWidth: Math.round(12 * Scaling.uiScale)
                        Layout.minimumWidth: Math.round(12 * Scaling.uiScale)
                        height: parent.height
                        checked: groupArea.row.active === true
                        onToggled: root.workspace.setDeviceGroupActive(groupArea.nodeId, checked)

                        indicator: Image {
                            anchors.centerIn: parent
                            source: groupActiveToggle.checked ? FontAwesome.icon("solid/toggle-on") : FontAwesome.icon("solid/toggle-off")
                            sourceSize: Qt.size(9.5 * Scaling.uiScale, 9.5 * Scaling.uiScale)
                            // dim when this node is on but an ancestor gates it off
                            opacity: (groupActiveToggle.checked && !groupArea.row.effectiveActive) ? 0.25 : 0.6
                        }

                        InfoToolTip {
                            textValue: "Toggle group active"
                        }
                    }

                    CheckBox {
                        id: groupRenderToggle
                        Layout.maximumWidth: Math.round(16 * Scaling.uiScale)
                        Layout.minimumWidth: Math.round(16 * Scaling.uiScale)
                        height: parent.height
                        checked: groupArea.row.render === true
                        onToggled: root.workspace.setDeviceGroupRender(groupArea.nodeId, checked)

                        indicator: Image {
                            anchors.centerIn: parent
                            source: groupRenderToggle.checked ? FontAwesome.icon("solid/eye") : FontAwesome.icon("solid/eye-slash")
                            sourceSize: Qt.size(9.5 * Scaling.uiScale, 9.5 * Scaling.uiScale)
                            opacity: (groupRenderToggle.checked && !groupArea.row.effectiveRender) ? 0.25 : 0.6
                        }

                        InfoToolTip {
                            textValue: "Toggle group rendering"
                        }
                    }

                    // label: editable on double-click, falls back to greyed id
                    Rectangle {
                        id: groupLabelBox
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "transparent"
                        border.color: groupLabelInput.activeFocus ? ThemeColors.highlight : "transparent"
                        border.width: 1

                        property bool editing: false
                        property string startText: ""

                        function commit() {
                            if (!editing)
                                return;
                            editing = false;
                            if (groupLabelInput.text !== startText)
                                root.workspace.setDeviceGroupLabel(groupArea.nodeId, groupLabelInput.text);
                        }

                        TextInput {
                            id: groupLabelInput
                            anchors.fill: parent
                            verticalAlignment: TextEdit.AlignVCenter
                            font: Scaling.uiFont
                            color: ThemeColors.text
                            selectionColor: ThemeColors.highlight
                            selectedTextColor: ThemeColors.highlightedText

                            enabled: groupLabelBox.editing
                            selectByMouse: groupLabelBox.editing

                            // reactive so it survives the Loader assigning row after construction
                            text: (groupArea.row.label && String(groupArea.row.label).length > 0) ? String(groupArea.row.label) : groupArea.nodeId
                            opacity: (groupArea.row.label && String(groupArea.row.label).length > 0) ? 1.0 : 0.4

                            onEditingFinished: groupLabelBox.commit()
                            onActiveFocusChanged: if (!activeFocus && groupLabelBox.editing)
                                groupLabelBox.commit()
                        }
                    }
                }
            }
        }
    }

    //
    // DEVICE DELEGATE
    //

    Component {
        id: deviceDelegate

        Item {
            id: dragArea

            property var row: ({})
            property int rowIndex: -1
            property int deviceIndex: row.deviceIndex !== undefined ? row.deviceIndex : -1
            property var modelData: deviceIndex >= 0 && root.workspace ? root.workspace.deviceAdapters[deviceIndex] : null
            property string nodeId: row.id !== undefined ? String(row.id) : ""
            property string nodeKind: "device"
            property real indent: (row.depth || 0) * root.indentStep
            readonly property real rowHeight: root.deviceRowHeight

            width: list.width
            height: rowHeight
            z: deviceDrag.active ? 1 : 0

            Rectangle {
                id: content

                property bool selected: root.workspace && root.workspace.selectedDeviceIndex === dragArea.deviceIndex
                property bool hovered: hoverHandler.hovered

                HoverHandler {
                    id: hoverHandler
                }

                width: list.width
                height: dragArea.rowHeight

                color: hovered ? ThemeColors.midlight : selected ? ThemeColors.mid : ThemeColors.almostdark
                border.width: 0

                states: State {
                    when: deviceDrag.active
                    ParentChange {
                        target: content
                        parent: dragLayer
                        x: 0
                    }
                }

                DragHandler {
                    id: deviceDrag
                    target: null
                    enabled: !deviceLabelBox.editing
                    xAxis.enabled: false
                    yAxis.enabled: true
                    onActiveChanged: {
                        if (active)
                            content.y = dragArea.mapToItem(dragLayer, 0, 0).y;
                        else
                            root.finishDrag(dragArea.nodeId, dragArea.nodeKind);
                    }
                    onCentroidChanged: {
                        if (active) {
                            content.y = dragArea.mapToItem(dragLayer, 0, 0).y + activeTranslation.y;
                            root.updateDragTarget(centroid.scenePosition.x, centroid.scenePosition.y, dragArea.nodeId);
                        }
                    }
                }

                TapHandler {
                    id: deviceTap
                    enabled: !deviceLabelBox.editing
                    acceptedButtons: Qt.LeftButton | Qt.RightButton
                    onTapped: (point, button) => {
                        list.forceActiveFocus();
                        if (button === Qt.LeftButton)
                            setSelectedIndex(dragArea.rowIndex, dragArea.deviceIndex);
                    }
                    onDoubleTapped: (point, button) => {
                        if (button !== Qt.LeftButton || !dragArea.modelData)
                            return;
                        const p = deviceLabelBox.mapFromItem(content, point.position.x, point.position.y);
                        if (p.x >= 0 && p.x <= deviceLabelBox.width && p.y >= 0 && p.y <= deviceLabelBox.height) {
                            deviceLabelBox.editing = true;
                            deviceLabelBox.startText = deviceLabelInput.text;
                            deviceLabelInput.forceActiveFocus();
                            deviceLabelInput.selectAll();
                        }
                    }
                }

                Rectangle {
                    anchors {
                        left: parent.left
                        right: parent.right
                        top: parent.top
                    }
                    height: Math.max(2, Math.round(2 * Scaling.uiScale))
                    color: ThemeColors.highlight
                    visible: root.dragOverIndex === dragArea.rowIndex && root.dragOverZone === "before"
                }
                Rectangle {
                    anchors {
                        left: parent.left
                        right: parent.right
                        bottom: parent.bottom
                    }
                    height: Math.max(2, Math.round(2 * Scaling.uiScale))
                    color: ThemeColors.highlight
                    visible: root.dragOverIndex === dragArea.rowIndex && root.dragOverZone === "after"
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: Math.round(8 * Scaling.uiScale) + dragArea.indent
                    anchors.rightMargin: Math.round(8 * Scaling.uiScale)
                    spacing: Math.round(8 * Scaling.uiScale)

                    CheckBox {
                        id: activeToggle
                        checked: dragArea.modelData ? dragArea.modelData.active : false
                        onToggled: if (dragArea.modelData)
                            dragArea.modelData.active = checked
                        Layout.maximumWidth: Math.round(16 * Scaling.uiScale)
                        Layout.minimumWidth: Math.round(16 * Scaling.uiScale)
                        height: parent.height

                        indicator: Image {
                            anchors.centerIn: parent
                            source: activeToggle.checked ? FontAwesome.icon("solid/toggle-on") : FontAwesome.icon("solid/toggle-off")
                            sourceSize: Qt.size(9.5 * Scaling.uiScale, 9.5 * Scaling.uiScale)
                            opacity: (activeToggle.checked && !dragArea.row.effectiveActive) ? 0.25 : (activeToggle.hovered ? 0.7 : 0.5)
                        }

                        InfoToolTip {
                            textValue: "Toggle device active"
                        }
                    }

                    CheckBox {
                        id: renderToggle
                        checked: dragArea.modelData ? dragArea.modelData.render : false
                        onToggled: if (dragArea.modelData)
                            dragArea.modelData.render = checked
                        Layout.maximumWidth: Math.round(16 * Scaling.uiScale)
                        Layout.minimumWidth: Math.round(16 * Scaling.uiScale)
                        height: parent.height

                        indicator: Image {
                            anchors.centerIn: parent
                            source: renderToggle.checked ? FontAwesome.icon("solid/eye") : FontAwesome.icon("solid/eye-slash")
                            sourceSize: Qt.size(9.5 * Scaling.uiScale, 9.5 * Scaling.uiScale)
                            opacity: (renderToggle.checked && !dragArea.row.effectiveRender) ? 0.25 : (renderToggle.hovered ? 0.7 : 0.5)
                        }

                        InfoToolTip {
                            textValue: "Toggle device rendering"
                        }
                    }

                    Rectangle {
                        id: statusCircle
                        width: 7 * Scaling.uiScale
                        height: 7 * Scaling.uiScale
                        radius: 3.5 * Scaling.uiScale
                        color: {
                            const m = dragArea.modelData;
                            if (!m || m.status === undefined)
                                return ThemeColors.inactive;
                            if (m.pluginNullState)
                                return ThemeColors.error;
                            switch (m.status) {
                            case UiEnums.WorkspaceDeviceStatus.Loaded:
                                return ThemeColors.neutralSuccess;
                            case UiEnums.WorkspaceDeviceStatus.Active:
                                return ThemeColors.success;
                            case UiEnums.WorkspaceDeviceStatus.Missing:
                                return ThemeColors.error;
                            default:
                                return ThemeColors.inactive;
                            }
                        }
                    }

                    // label: editable on double-click, falls back to greyed id
                    Rectangle {
                        id: deviceLabelBox
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "transparent"
                        border.color: deviceLabelInput.activeFocus ? ThemeColors.highlight : "transparent"
                        border.width: 1

                        property bool editing: false
                        property string startText: ""

                        function commit() {
                            if (!editing)
                                return;
                            editing = false;
                            if (dragArea.modelData && deviceLabelInput.text !== startText)
                                dragArea.modelData.label = deviceLabelInput.text;
                        }

                        TextInput {
                            id: deviceLabelInput
                            anchors.fill: parent
                            verticalAlignment: TextEdit.AlignVCenter
                            font: Scaling.uiFont
                            color: ThemeColors.text
                            selectionColor: ThemeColors.highlight
                            selectedTextColor: ThemeColors.highlightedText

                            enabled: deviceLabelBox.editing
                            selectByMouse: deviceLabelBox.editing

                            text: {
                                if (!dragArea.modelData)
                                    return "";
                                const lbl = dragArea.modelData.label;
                                return (lbl && String(lbl).length > 0) ? String(lbl) : String(dragArea.modelData.id);
                            }
                            opacity: (dragArea.modelData && dragArea.modelData.label && String(dragArea.modelData.label).length > 0) ? 1.0 : 0.5

                            onEditingFinished: deviceLabelBox.commit()
                            onActiveFocusChanged: if (!activeFocus && deviceLabelBox.editing)
                                deviceLabelBox.commit()
                        }
                    }

                    Text {
                        id: deviceTypeText
                        elide: Text.ElideRight
                        text: dragArea.modelData ? (dragArea.modelData.displayName() + (dragArea.modelData.pluginNullState ? " (Unloaded)" : "")) : ""
                        color: ThemeColors.text
                        opacity: dragArea.modelData && dragArea.modelData.pluginNullState ? .25 : .5
                        font: Scaling.uiFont
                        Layout.rightMargin: Scaling.uiScale * 2
                    }
                }
            }
        }
    }

    Rectangle {
        id: frame
        anchors.fill: parent
        color: ThemeColors.alternateBase
        border.color: root.dropToRoot ? ThemeColors.highlight : ThemeColors.mid
        border.width: 1
        clip: true

        ListView {
            id: list
            anchors.fill: parent
            anchors.margins: 1

            model: root.workspace ? root.workspace.deviceTreeRows : []
            focus: true
            activeFocusOnTab: true
            keyNavigationEnabled: true
            // disabled so per-row vertical DragHandlers don't fight the flick;
            // scrollbar + wheel still scroll
            interactive: false
            boundsBehavior: Flickable.StopAtBounds

            ScrollBar.vertical: ScrollBar {
                policy: ScrollBar.AsNeeded
            }

            WheelHandler {
                acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
                onWheel: event => {
                    if (list.contentHeight <= list.height)
                        return;
                    const dy = event.angleDelta.y !== 0 ? event.angleDelta.y : event.pixelDelta.y;
                    list.contentY = Math.max(0, Math.min(list.contentHeight - list.height, list.contentY - dy));
                }
            }

            // keyboard: up/down moves, enter/space selects a device row
            Keys.onPressed: event => {
                if (event.key === Qt.Key_Up) {
                    list.decrementCurrentIndex();
                    list.positionViewAtIndex(list.currentIndex, ListView.Contain);
                    event.accepted = true;
                } else if (event.key === Qt.Key_Down) {
                    list.incrementCurrentIndex();
                    list.positionViewAtIndex(list.currentIndex, ListView.Contain);
                    event.accepted = true;
                } else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter || event.key === Qt.Key_Space) {
                    if (list.currentIndex >= 0) {
                        const r = list.model[list.currentIndex];
                        if (r && r.kind === "device")
                            setSelectedIndex(list.currentIndex, r.deviceIndex);
                    }
                    event.accepted = true;
                }
            }

            delegate: rowLoader

            // right-click on empty list space creates a root-level group
            TapHandler {
                acceptedButtons: Qt.RightButton
                onTapped: {
                    contextMenu.targetGroupId = "";
                    contextMenu.onGroup = false;
                    contextMenu.popup();
                }
            }
        }

        // focus ring
        Rectangle {
            anchors.fill: parent
            anchors.margins: 1
            radius: frame.radius
            color: "transparent"
            border.width: list.activeFocus ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0
            border.color: ThemeColors.highlight
            visible: border.width > 0
        }

        Item {
            id: dragLayer
            anchors.fill: parent
            z: 1
            opacity: 0.5
        }
    }
}
