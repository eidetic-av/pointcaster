import QtQuick 2.6
import QtQuick.Controls 2.6
import "qrc:/kddockwidgets/qtquick/views/qml/" as KDDW

import Pointcaster 1.0

KDDW.TabBarBase {
    id: root

    readonly property real tabBarTextPixelSize: Math.round(14 * Scaling.uiScale)
    readonly property real tabBarVerticalPadding: Math.round(7 * Scaling.uiScale)
    readonly property real tabBarHorizontalPadding: Math.round(10 * Scaling.uiScale)
    height: tabBarTextPixelSize + tabBarVerticalPadding * 2
    implicitHeight: height

    currentTabIndex: 0

    // right click on tab context menu state
    property int contextTabIndex: -1

    // keyboard focus tracking:
    //   - 0..(tabCount-1) means a tab
    //   - tabCount means the "+" button
    property int keyboardFocusIndex: 0

    // the tab being renamed in place, or -1
    property int renamingIndex: -1

    // set from while a tab is dragged along the strip
    property int reorderTabIndex: -1
    property int reorderInsertIndex: -1

    // tabs in the central group are sessions, addressed by the label they show
    function sessionIdAtIndex(index) {
        const item = tabItemAt(index);
        if (!item || !workspaceModel)
            return "";
        return workspaceModel.sessionIdForLabel(String(item.tabTitle));
    }

    function requestRenameTab(index) {
        root.renamingIndex = index;
    }
    function requestDuplicateTab(index) {
        const sessionId = sessionIdAtIndex(index);
        if (sessionId.length > 0)
            workspaceModel.duplicateSession(sessionId);
    }
    function requestDeleteTab(index) {
        const sessionId = sessionIdAtIndex(index);
        if (sessionId.length > 0)
            workspaceModel.removeSession(sessionId);
    }
    function commitRename(index, text) {
        // losing focus and pressing return both land here
        if (root.renamingIndex !== index)
            return;
        root.renamingIndex = -1;
        const sessionId = sessionIdAtIndex(index);
        if (sessionId.length > 0)
            workspaceModel.setSessionLabel(sessionId, text);
    }

    Menu {
        id: tabContextMenu
        font: Scaling.uiFont

        Action {
            text: qsTr("Rename")
            enabled: root.contextTabIndex >= 0
            onTriggered: root.requestRenameTab(root.contextTabIndex)
        }

        Action {
            text: qsTr("Duplicate")
            enabled: root.contextTabIndex >= 0
            onTriggered: root.requestDuplicateTab(root.contextTabIndex)
        }

        MenuSeparator {}

        Action {
            text: qsTr("Delete")
            // a workspace always keeps one session
            enabled: root.contextTabIndex >= 0 && root.tabCount() > 1
            onTriggered: root.requestDeleteTab(root.contextTabIndex)
        }

        onAboutToHide: root.contextTabIndex = -1
    }

    Rectangle {
        id: tabBarBackground
        color: ThemeColors.middark
        anchors.fill: parent
        z: -1000
        enabled: false
    }

    function tabCount() {
        return (tabBarCpp && tabBarCpp.dockWidgetModel) ? tabBarCpp.dockWidgetModel.count : 0;
    }

    function clampKeyboardFocusIndex(index) {
        var maxIndexInclusive = tabCount(); // tabs + add button
        return Math.max(0, Math.min(maxIndexInclusive, index));
    }

    function selectTab(index) {
        if (!tabBarCpp)
            return;
        var count = tabCount();
        if (count <= 0)
            return;
        var clamped = Math.max(0, Math.min(count - 1, index));
        tabBarCpp.setCurrentIndex(clamped);
    }

    function activateAddButton() {
        if (workspaceModel)
            workspaceModel.addSession();
    }

    function tabItemAt(index) {
        return tabRepeater.itemAt(index);
    }

    function focusTabItem(index) {
        var item = tabItemAt(index);
        if (item)
            item.forceActiveFocus();
    }

    function focusAddButtonItem() {
        addButtonFocusScope.forceActiveFocus();
    }

    function moveKeyboardFocus(delta) {
        var next = clampKeyboardFocusIndex(root.keyboardFocusIndex + delta);
        root.keyboardFocusIndex = next;

        // 0..tabCount-1 => tab, tabCount => add button
        if (next < tabCount()) {
            focusTabItem(next);
        } else {
            focusAddButtonItem();
        }
    }

    function activateKeyboardFocus() {
        if (root.keyboardFocusIndex < tabCount()) {
            selectTab(root.keyboardFocusIndex);
            focusTabItem(root.keyboardFocusIndex);
        } else {
            activateAddButton();
            focusAddButtonItem();
        }
    }

    /// Required by KDDW C++ for dragging/reordering/floating.
    function getTabAtIndex(index) {
        return tabItemAt(index);
    }

    /// Required by KDDW C++ for hit-testing which tab is under the cursor.
    function getTabIndexAtPosition(globalPoint) {
        var count = tabCount();
        for (var i = 0; i < count; ++i) {
            var tab = tabItemAt(i);
            if (!tab)
                continue;
            var localPt = tab.mapFromGlobal(globalPoint.x, globalPoint.y);
            if (tab.contains(localPt))
                return i;
        }
        return -1;
    }

    Row {
        id: tabBarRow

        // keep our visuals above the built-in mouse layer, but dont steal left click/drag
        z: root.mouseAreaZ + 1
        anchors.fill: parent
        spacing: 0

        // hover tracking for colour changes
        property int hoveredIndex: -1

        Repeater {
            id: tabRepeater
            model: root.groupCpp ? root.groupCpp.tabBar.dockWidgetModel : 0

            FocusScope {
                id: tab
                height: tabBarRow.height

                // ---- content-sized width knobs ----
                readonly property int tabHorizontalPadding: root.tabBarHorizontalPadding
                readonly property int tabMinWidth: Math.round(56 * Scaling.uiScale)
                readonly property int tabMaxWidth: Math.round(220 * Scaling.uiScale)

                implicitWidth: Math.max(tabMinWidth, Math.min(tabMaxWidth, titleText.implicitWidth + tabHorizontalPadding * 2))
                width: implicitWidth
                // -----------------------------------

                readonly property int tabIndex: index
                readonly property string tabTitle: title
                readonly property bool isRenaming: root.renamingIndex === tabIndex

                opacity: root.reorderTabIndex === tabIndex ? 0.4 : 1

                readonly property bool isCurrent: tabIndex === root.groupCpp.currentIndex
                readonly property bool isHovered: tabIndex === tabBarRow.hoveredIndex
                readonly property bool isKeyboardFocused: root.keyboardFocusIndex === tabIndex

                activeFocusOnTab: true
                focus: isKeyboardFocused

                Accessible.role: Accessible.PageTab
                Accessible.name: title
                Accessible.focusable: true
                Accessible.checked: isCurrent

                Keys.onPressed: event => {
                    switch (event.key) {
                    case Qt.Key_Left:
                        root.moveKeyboardFocus(-1);
                        event.accepted = true;
                        break;
                    case Qt.Key_Right:
                        root.moveKeyboardFocus(+1);
                        event.accepted = true;
                        break;
                    case Qt.Key_Home:
                        root.keyboardFocusIndex = 0;
                        root.focusTabItem(0);
                        event.accepted = true;
                        break;
                    case Qt.Key_End:
                        // End goes to "+" (common in tab strips), not last tab
                        root.keyboardFocusIndex = root.tabCount();
                        root.focusAddButtonItem();
                        event.accepted = true;
                        break;
                    case Qt.Key_Return:
                    case Qt.Key_Enter:
                    case Qt.Key_Space:
                        root.activateKeyboardFocus();
                        event.accepted = true;
                        break;
                    default:
                        break;
                    }
                }

                // background fill
                Rectangle {
                    anchors.fill: parent
                    color: tab.isHovered ? ThemeColors.midlight : (tab.isCurrent ? ThemeColors.mid : ThemeColors.middark)
                }

                // 1px "active session" line
                Rectangle {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    height: tab.isCurrent ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0
                    color: ThemeColors.highlight
                    z: 10
                }

                Item {
                    id: textSlot
                    anchors.fill: parent
                    anchors.leftMargin: tab.tabHorizontalPadding
                    anchors.rightMargin: tab.tabHorizontalPadding
                    z: 20

                    Text {
                        id: titleText
                        anchors.centerIn: parent
                        width: textSlot.width
                        horizontalAlignment: Text.AlignHCenter
                        visible: !tab.isRenaming

                        text: tab.tabTitle
                        font: Scaling.uiFont
                        color: ThemeColors.text
                        elide: Text.ElideRight
                    }

                    TextInput {
                        id: titleEditor
                        anchors.fill: parent
                        visible: tab.isRenaming
                        enabled: tab.isRenaming
                        verticalAlignment: TextInput.AlignVCenter
                        horizontalAlignment: TextInput.AlignHCenter
                        font: Scaling.uiFont
                        color: ThemeColors.text
                        selectionColor: ThemeColors.highlight
                        selectedTextColor: ThemeColors.highlightedText
                        selectByMouse: true

                        onVisibleChanged: {
                            if (!visible)
                                return;
                            text = tab.tabTitle;
                            forceActiveFocus();
                            selectAll();
                        }

                        Keys.onEscapePressed: root.renamingIndex = -1
                        onEditingFinished: root.commitRename(tab.tabIndex, text)
                        onActiveFocusChanged: if (!activeFocus && tab.isRenaming)
                            root.commitRename(tab.tabIndex, text)
                    }
                }

                // keyboard focus ring (separate from "active" line)
                Rectangle {
                    anchors.fill: parent
                    color: "transparent"
                    border.width: tab.activeFocus ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0
                    border.color: ThemeColors.highlight
                    z: 30
                }

                // mouse area for click actions on tabs

                MouseArea {
                    // only for session tabs though... since the right click context actions
                    // are all session related for now
                    enabled: root.groupCpp.isCentralGroup
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.RightButton | Qt.LeftButton
                    z: root.mouseAreaZ + 2

                    onEntered: tabBarRow.hoveredIndex = tab.tabIndex
                    onExited: {
                        if (tabBarRow.hoveredIndex === tab.tabIndex)
                            tabBarRow.hoveredIndex = -1;
                    }

                    onClicked: mouse => {
                        if (mouse.button == Qt.LeftButton) {
                            root.selectTab(tab.tabIndex);
                            return;
                        }
                        // handle right clicks
                        root.contextTabIndex = tab.tabIndex;
                        var p = tab.mapToItem(root, mouse.x, mouse.y);
                        tabContextMenu.popup(Qt.point(p.x, p.y));
                    }

                    onDoubleClicked: mouse => {
                        if (mouse.button == Qt.RightButton) return;
                        // handle double clicks for rename of session tabs
                        root.requestRenameTab(tab.tabIndex)
                    }
                }
            }
        }

        // only show the 'add session' button for the central group of session views
        Loader {
            active: root.groupCpp !== null ? root.groupCpp.isCentralGroup : false

            // when visible its wrapped in a FocusScope to be tabbable for navigation
            sourceComponent: FocusScope {
                id: addButtonFocusScope
                width: Math.round(34 * Scaling.uiScale)
                height: tabBarRow.height

                readonly property bool isHovered: addButtonMouseArea.containsMouse
                readonly property bool isKeyboardFocused: root.keyboardFocusIndex === root.tabCount()

                activeFocusOnTab: true
                focus: isKeyboardFocused

                Rectangle {
                    anchors.fill: parent
                    color: ThemeColors.middark
                }

                // mimic hover highlight on keyboard focus as well
                Rectangle {
                    anchors.fill: parent
                    color: "transparent"
                    border.width: addButtonFocusScope.activeFocus ? Math.max(1, Math.round(1 * Scaling.uiScale)) : 0
                    border.color: ThemeColors.highlight
                }

                InfoToolTip {
                    visible: addButtonMouseArea.containsMouse
                    textValue: "Add new session"
                }

                Image {
                    anchors.centerIn: parent
                    opacity: (addButtonMouseArea.containsMouse || addButtonFocusScope.activeFocus) ? 1 : 0.5
                    width: Math.round(14 * Scaling.uiScale)
                    height: Math.round(14 * Scaling.uiScale)
                    fillMode: Image.PreserveAspectFit
                    source: FontAwesome.icon('solid/plus')
                }

                Keys.onPressed: event => {
                    switch (event.key) {
                    case Qt.Key_Left:
                        root.moveKeyboardFocus(-1);
                        event.accepted = true;
                        break;
                    case Qt.Key_Right:
                        root.moveKeyboardFocus(+1);
                        event.accepted = true;
                        break;
                    case Qt.Key_Home:
                        root.keyboardFocusIndex = 0;
                        root.focusTabItem(0);
                        event.accepted = true;
                        break;
                    case Qt.Key_End:
                        // stay here
                        event.accepted = true;
                        break;
                    case Qt.Key_Return:
                    case Qt.Key_Enter:
                    case Qt.Key_Space:
                        root.activateKeyboardFocus();
                        event.accepted = true;
                        break;
                    default:
                        break;
                    }
                }

                MouseArea {
                    id: addButtonMouseArea
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.LeftButton
                    onEntered: tabBarRow.hoveredIndex = -1
                    onClicked: {
                        root.keyboardFocusIndex = root.tabCount();
                        addButtonFocusScope.forceActiveFocus();
                        root.activateAddButton();
                    }
                }
            }
        }

        Connections {
            target: tabBarCpp
            function onHoveredTabIndexChanged(index) {
                tabBarRow.hoveredIndex = index;
            }
        }
    }

    TabDropIndicator {
        id: reorderIndicator

        anchors.top: parent.top
        anchors.bottom: parent.bottom
        x: 0
        width: root.width
        z: root.mouseAreaZ + 10

        visible: root.reorderInsertIndex >= 0
        showBackground: false
        accent: ThemeColors.highlight

        dropLineX: {
            const tabsInBar = root.tabCount();
            if (root.reorderInsertIndex <= 0)
                return 0;
            if (root.reorderInsertIndex >= tabsInBar) {
                const lastTab = root.tabItemAt(tabsInBar - 1);
                return lastTab ? lastTab.x + lastTab.width : 0;
            }
            const nextTab = root.tabItemAt(root.reorderInsertIndex);
            return nextTab ? nextTab.x : 0;
        }
    }

    // Initialise keyboard focus to current tab when created or when selection changes.
    Component.onCompleted: {
        root.keyboardFocusIndex = Math.max(0, root.groupCpp ? root.groupCpp.currentIndex : 0);
        root.focusTabItem(root.keyboardFocusIndex);
    }

    Connections {
        target: root.groupCpp
        function onCurrentIndexChanged() {
            root.keyboardFocusIndex = Math.max(0, root.groupCpp.currentIndex);
            root.focusTabItem(root.keyboardFocusIndex);
        }
    }
}
