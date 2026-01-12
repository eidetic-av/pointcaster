import QtQuick 2.6
import QtQuick.Controls 2.6
import "qrc:/kddockwidgets/qtquick/views/qml/" as KDDW

KDDW.TabBarBase {
    id: root

    implicitHeight: 30
    currentTabIndex: 0

    // right click on tab context menu state
    property int contextTabIndex: -1

    // keyboard focus tracking:
    //   - 0..(tabCount-1) means a tab
    //   - tabCount means the "+" button
    property int keyboardFocusIndex: 0

    // TODO: wire these later to real behaviour
    function requestRenameTab(index)    { console.log("rename tab", index) }
    function requestDuplicateTab(index) { console.log("duplicate tab", index) }
    function requestDeleteTab(index)    { console.log("delete tab", index) }

    Menu {
        id: tabContextMenu

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

        MenuSeparator { }

        Action {
            text: qsTr("Delete")
            enabled: root.contextTabIndex >= 0
            onTriggered: root.requestDeleteTab(root.contextTabIndex)
        }

        onAboutToHide: root.contextTabIndex = -1
    }

    // Background: must not sit on top of tabs, or it will eat hover/click.
    Rectangle {
        id: tabBarBackground
        color: DarkPalette.middark
        anchors.fill: parent
        z: -1000
        enabled: false
    }

    function tabCount() {
        return (tabBarCpp && tabBarCpp.dockWidgetModel) ? tabBarCpp.dockWidgetModel.count : 0
    }

    function clampKeyboardFocusIndex(index) {
        var maxIndexInclusive = tabCount() // tabs + add button
        return Math.max(0, Math.min(maxIndexInclusive, index))
    }

    function selectTab(index) {
        if (!tabBarCpp) return
        var count = tabCount()
        if (count <= 0) return

        var clamped = Math.max(0, Math.min(count - 1, index))
        tabBarCpp.setCurrentIndex(clamped)
    }

    function activateAddButton() {
        if (!tabBarCpp) return

        var uniqueName = Date.now()
        var code = `import com.kdab.dockwidgets 2.0 as KDDW;
                    import QtQuick 2.6;
                    KDDW.DockWidget {
                        uniqueName: "${uniqueName}";
                        title: "dynamic";
                        affinities: ["view"]
                        Rectangle { color: "#85baa1"; anchors.fill: parent; }
                    }`

        var newDW = Qt.createQmlObject(code, root)
        tabBarCpp.addDockWidgetAsTab(newDW)
    }

    function focusTabItem(index) {
        var item = tabBarRow.tabItemAt(index)
        if (item) item.forceActiveFocus()
    }

    function focusAddButtonItem() {
        addButtonFocusScope.forceActiveFocus()
    }

    function moveKeyboardFocus(delta) {
        var next = clampKeyboardFocusIndex(root.keyboardFocusIndex + delta)
        root.keyboardFocusIndex = next

        // 0..tabCount-1 => tab, tabCount => add button
        if (next < tabCount()) {
            focusTabItem(next)
        } else {
            focusAddButtonItem()
        }
    }

    function activateKeyboardFocus() {
        if (root.keyboardFocusIndex < tabCount()) {
            selectTab(root.keyboardFocusIndex)
            focusTabItem(root.keyboardFocusIndex)
        } else {
            activateAddButton()
            focusAddButtonItem()
        }
    }

    Row {
        id: tabBarRow

        z: root.mouseAreaZ ? (root.mouseAreaZ.z + 1) : 1
        anchors.fill: parent
        spacing: 0

        // hover tracking for colour changes
        property int hoveredIndex: -1

        // keep refs to delegate items so we don't rely on Row.children ordering
        property var tabItemsByIndex: ({})
        function registerTabItem(index, item) { tabItemsByIndex[index] = item }
        function unregisterTabItem(index, item) {
            if (tabItemsByIndex[index] === item) delete tabItemsByIndex[index]
        }
        function tabItemAt(index) { return tabItemsByIndex[index] }

        Repeater {
            id: tabRepeater
            model: root.groupCpp ? root.groupCpp.tabBar.dockWidgetModel : 0

            FocusScope {
                id: tab
                height: tabBarRow.height
                width: 120

                readonly property int tabIndex: index
                readonly property bool isCurrent: tabIndex === root.groupCpp.currentIndex
                readonly property bool isHovered: tabIndex === tabBarRow.hoveredIndex
                readonly property bool isKeyboardFocused: root.keyboardFocusIndex === tabIndex

                Component.onCompleted: tabBarRow.registerTabItem(tabIndex, tab)
                Component.onDestruction: tabBarRow.unregisterTabItem(tabIndex, tab)

                // "Roving tabindex": only the *keyboard focused* item participates in Tab focus chain
                // (this lets Tab enter/leave the bar without stepping through every tab)
                activeFocusOnTab: isKeyboardFocused
                focus: isKeyboardFocused

                Accessible.role: Accessible.PageTab
                Accessible.name: title
                Accessible.focusable: true
                Accessible.checked: isCurrent

                function activate() {
                    root.keyboardFocusIndex = tabIndex
                    root.selectTab(tabIndex)
                    forceActiveFocus()
                }

                Keys.onPressed: (event) => {
                    switch (event.key) {
                    case Qt.Key_Left:
                        root.moveKeyboardFocus(-1)
                        event.accepted = true
                        break
                    case Qt.Key_Right:
                        root.moveKeyboardFocus(+1)
                        event.accepted = true
                        break
                    case Qt.Key_Home:
                        root.keyboardFocusIndex = 0
                        root.focusTabItem(0)
                        event.accepted = true
                        break
                    case Qt.Key_End:
                        // End goes to "+" (common in tab strips), not last tab
                        root.keyboardFocusIndex = root.tabCount()
                        root.focusAddButtonItem()
                        event.accepted = true
                        break
                    case Qt.Key_Return:
                    case Qt.Key_Enter:
                    case Qt.Key_Space:
                        root.activateKeyboardFocus()
                        event.accepted = true
                        break
                    default:
                        break
                    }
                }

                // background fill
                Rectangle {
                    anchors.fill: parent
                    color: tab.isHovered
                           ? DarkPalette.midlight
                           : (tab.isCurrent ? DarkPalette.mid : DarkPalette.middark)
                }

                // 1px "active" line
                Rectangle {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    height: tab.isCurrent ? 1 : 0
                    color: DarkPalette.highlight
                    z: 10
                }

                Text {
                    anchors.centerIn: parent
                    text: title
                    font.pixelSize: 15
                    color: DarkPalette.text
                    opacity: tab.isHovered ? 1 : tab.isCurrent ? 1 : 0.5
                    elide: Text.ElideRight
                    z: 20
                }

                // keyboard focus ring (separate from "active" line)
                Rectangle {
                    anchors.fill: parent
                    color: "transparent"
                    border.width: tab.activeFocus ? 1 : 0
                    border.color: DarkPalette.highlight
                    z: 30
                }

                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.LeftButton | Qt.RightButton
                    z: 100

                    onEntered: tabBarRow.hoveredIndex = tab.tabIndex
                    onExited:  if (tabBarRow.hoveredIndex === tab.tabIndex) tabBarRow.hoveredIndex = -1

                    onClicked: (mouse) => {
                        if (mouse.button === Qt.LeftButton) {
                            tab.activate()
                        } else if (mouse.button === Qt.RightButton) {
                            root.keyboardFocusIndex = tab.tabIndex
                            tab.forceActiveFocus()

                            root.contextTabIndex = tab.tabIndex
                            var p = tab.mapToItem(root, mouse.x, mouse.y)
                            tabContextMenu.popup(Qt.point(p.x, p.y))
                        }
                    }
                }
            }
        }

        // "+" add button as a focusable item in the same arrow-key cycle
        FocusScope {
            id: addButtonFocusScope
            width: 40
            height: tabBarRow.height

            readonly property bool isHovered: addButtonMouseArea.containsMouse
            readonly property bool isKeyboardFocused: root.keyboardFocusIndex === root.tabCount()

            activeFocusOnTab: isKeyboardFocused
            focus: isKeyboardFocused

            Accessible.role: Accessible.Button
            Accessible.name: "Add new session"
            Accessible.focusable: true

            Rectangle {
                anchors.fill: parent
                color: DarkPalette.middark
            }

            // mimic hover highlight on keyboard focus as well
            Rectangle {
                anchors.fill: parent
                color: "transparent"
                border.width: addButtonFocusScope.activeFocus ? 1 : 0
                border.color: DarkPalette.highlight
            }

            ToolTip.visible: addButtonMouseArea.containsMouse
            ToolTip.text: "Add new session"
            ToolTip.delay: 500

            Image {
                anchors.centerIn: parent
                opacity: (addButtonMouseArea.containsMouse || addButtonFocusScope.activeFocus) ? 1 : 0.5
                width: 16
                height: 16
                fillMode: Image.PreserveAspectFit
                source: FontAwesome.icon('solid/plus')
            }

            Keys.onPressed: (event) => {
                switch (event.key) {
                case Qt.Key_Left:
                    root.moveKeyboardFocus(-1)
                    event.accepted = true
                    break
                case Qt.Key_Right:
                    root.moveKeyboardFocus(+1)
                    event.accepted = true
                    break
                case Qt.Key_Home:
                    root.keyboardFocusIndex = 0
                    root.focusTabItem(0)
                    event.accepted = true
                    break
                case Qt.Key_End:
                    // stay here
                    event.accepted = true
                    break
                case Qt.Key_Return:
                case Qt.Key_Enter:
                case Qt.Key_Space:
                    root.activateKeyboardFocus()
                    event.accepted = true
                    break
                default:
                    break
                }
            }

            MouseArea {
                id: addButtonMouseArea
                anchors.fill: parent
                hoverEnabled: true
                acceptedButtons: Qt.LeftButton
                onEntered: tabBarRow.hoveredIndex = -1
                onClicked: {
                    root.keyboardFocusIndex = root.tabCount()
                    addButtonFocusScope.forceActiveFocus()
                    root.activateAddButton()
                }
            }
        }

        Connections {
            target: tabBarCpp
            function onHoveredTabIndexChanged(index) {
                tabBarRow.hoveredIndex = index
            }
        }
    }

    // Initialise keyboard focus to current tab when created / when selection changes.
    Component.onCompleted: {
        root.keyboardFocusIndex = Math.max(0, root.groupCpp ? root.groupCpp.currentIndex : 0)
        root.focusTabItem(root.keyboardFocusIndex)
    }

    Connections {
        target: root.groupCpp
        function onCurrentIndexChanged() {
            // keep keyboard focus following selection when selection changes externally
            root.keyboardFocusIndex = Math.max(0, root.groupCpp.currentIndex)
        }
    }
}
