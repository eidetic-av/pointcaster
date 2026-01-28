import QtQuick 2.6
import QtQuick.Controls 2.6
import "qrc:/kddockwidgets/qtquick/views/qml/" as KDDW

import Pointcaster 1.0

KDDW.TabBarBase {
    id: root

    readonly property real tabBarTextPixelSize: Math.round(14 * Scaling.uiScale)
    readonly property real tabBarVerticalPadding: Math.round(8 * Scaling.uiScale)
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

    // TODO: wire these later to real behaviour
    function requestRenameTab(index) {
        console.log("rename tab", index);
    }
    function requestDuplicateTab(index) {
        console.log("duplicate tab", index);
    }
    function requestDeleteTab(index) {
        console.log("delete tab", index);
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
            enabled: root.contextTabIndex >= 0
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
        if (!tabBarCpp)
            return;
        var uniqueName = Date.now();
        var code = `import com.kdab.dockwidgets 2.0 as KDDW;
                    import QtQuick 2.6;
                    KDDW.DockWidget {
                        uniqueName: "${uniqueName}";
                        title: "dynamic";
                        affinities: ["view"]
                        Rectangle { color: "#85baa1"; anchors.fill: parent; }
                    }`;

        var newDW = Qt.createQmlObject(code, root);
        tabBarCpp.addDockWidgetAsTab(newDW);
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
        z: root.mouseAreaZ ? (root.mouseAreaZ.z + 1) : 1
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

                readonly property bool isCurrent: tabIndex === root.groupCpp.currentIndex
                readonly property bool isHovered: tabIndex === tabBarRow.hoveredIndex
                readonly property bool isKeyboardFocused: root.keyboardFocusIndex === tabIndex

                activeFocusOnTab: isKeyboardFocused
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

                        text: title
                        font: Scaling.uiFont
                        color: ThemeColors.text
                        opacity: tab.isHovered ? 1 : tab.isCurrent ? 1 : 0.5
                        elide: Text.ElideRight
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

                // mouse area for a right click context menu on the tabs
                MouseArea {
                    // only for session tabs though... since the right click context actions
                    // are all session related for now
                    enabled: root.groupCpp.isCentralGroup
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.RightButton
                    z: root.mouseAreaZ ? (root.mouseAreaZ.z + 2) : 100

                    onEntered: tabBarRow.hoveredIndex = tab.tabIndex
                    onExited: {
                        if (tabBarRow.hoveredIndex === tab.tabIndex)
                            tabBarRow.hoveredIndex = -1;
                    }

                    onClicked: mouse => {
                        root.keyboardFocusIndex = tab.tabIndex;
                        tab.forceActiveFocus();

                        root.contextTabIndex = tab.tabIndex;
                        var p = tab.mapToItem(root, mouse.x, mouse.y);
                        tabContextMenu.popup(Qt.point(p.x, p.y));
                    }
                }

                Component.onCompleted: {
                    console.log("dockWidgetModel keys:", Object.keys(modelData));
                    console.log("title:", modelData.title, "affinities:", modelData.affinities);
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

                activeFocusOnTab: isKeyboardFocused
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
