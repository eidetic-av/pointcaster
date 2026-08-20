import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

Column {
    id: root

    required property var workspace

    required property var operators
    required property var pipelineAdapter

    // config path of the device or session hosting this pipeline, like
    // "device/group_a/cam_1"
    required property string hostPath

    signal addOperatorRequested(string operatorType)
    signal removeOperatorRequested(int operatorIndex)

    // fold keys are config paths
    readonly property string sectionFoldKey: hostPath ? hostPath + "/operators" : ""

    readonly property bool sectionExpanded: {
        if (!sectionFoldKey)
            return true;
        const storedValue = workspace.foldedPropertyPaths[sectionFoldKey];
        return storedValue === undefined ? true : storedValue;
    }

    spacing: 0

    readonly property int operatorCount: operators.length

    Rectangle {
        id: sectionHeader

        width: root.width
        height: Math.round(28 * Scaling.uiScale)
        color: sectionHeaderMouse.containsMouse ? ThemeColors.middark : (root.sectionExpanded ? ThemeColors.almostdark : ThemeColors.dark)
        border.color: ThemeColors.almostdark
        border.width: root.sectionExpanded ? 0 : Math.max(1, Math.round(1 * Scaling.uiScale))

        MouseArea {
            id: sectionHeaderMouse
            anchors.fill: parent
            hoverEnabled: true
            onClicked: {
                if (root.sectionFoldKey)
                    root.workspace.setFoldedProperty(root.sectionFoldKey, !root.sectionExpanded);
            }
        }

        Row {
            anchors {
                left: parent.left
                leftMargin: Math.round(5 * Scaling.uiScale)
                verticalCenter: parent.verticalCenter
            }
            spacing: Math.round(5 * Scaling.uiScale)

            Image {
                anchors.verticalCenter: parent.verticalCenter
                width: Math.round(12 * Scaling.uiScale)
                fillMode: Image.PreserveAspectFit
                source: root.sectionExpanded ? FontAwesome.icon("solid/caret-down") : FontAwesome.icon("solid/caret-right")
                opacity: 0.75
            }

            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: "Operators"
                font: Scaling.uiFont
                color: ThemeColors.text
            }
        }
    }

    Rectangle {
        id: pipelineBody

        visible: root.sectionExpanded
        width: root.width
        height: visible ? pipelineColumn.height : 0
        color: ThemeColors.window
        border.color: ThemeColors.almostdark
        border.width: Math.max(1, Math.round(1 * Scaling.uiScale))

        Column {
            id: pipelineColumn
            width: parent.width
            topPadding: Math.round(4 * Scaling.uiScale)
            bottomPadding: Math.round(8 * Scaling.uiScale)

            Item {
                id: inputLabel
                width: parent.width
                height: Math.round(18 * Scaling.uiScale)

                Row {
                    anchors.centerIn: parent
                    spacing: Math.round(4 * Scaling.uiScale)

                    Rectangle {
                        width: Math.round(6 * Scaling.uiScale)
                        height: width
                        radius: width / 2
                        color: ThemeColors.success
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    Text {
                        text: "INPUT"
                        font: Scaling.uiSmallFont
                        color: ThemeColors.placeholderText
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }

            Repeater {
                id: operatorRepeater
                model: root.operators

                Column {
                    id: operatorDelegate

                    required property var modelData
                    required property int index

                    width: pipelineColumn.width

                    Item {
                        id: pipeConnector
                        width: parent.width
                        height: Math.round(16 * Scaling.uiScale)

                        Rectangle {
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: Math.max(1, Math.round(2 * Scaling.uiScale))
                            height: parent.height
                            color: ThemeColors.accent
                            opacity: 0.4
                        }

                        Rectangle {
                            anchors.centerIn: parent
                            width: Math.round(8 * Scaling.uiScale)
                            height: width
                            radius: width / 2
                            color: ThemeColors.accent
                            border.width: Math.max(1, Math.round(2 * Scaling.uiScale))
                            border.color: ThemeColors.dark
                        }
                    }

                    Rectangle {
                        id: operatorContainer

                        // The inner ConfigAdapter (generated), accessed via the
                        // OperatorAdapter's configAdapter property.
                        property var configAdapter: modelData ? modelData.configAdapter : null
                        property bool selected: modelData && root.workspace.selectedOperatorAdapter === modelData
                        property bool active: configAdapter ? configAdapter.active : true
                        property string operatorId: configAdapter ? String(configAdapter.value("id")) : "unknown"

                        readonly property string operatorLabel: configAdapter ? String(configAdapter.label) : ""
                        readonly property string displayName: operatorLabel.length > 0 ? operatorLabel : operatorId

                        // the operator's own config path, like
                        // "device/group_a/cam_1/operators/<id>"
                        readonly property string foldKey: configAdapter && configAdapter.configPath ? String(configAdapter.configPath) : ""

                        // operators start folded, and stay wherever the user
                        // last put them once the fold map has an entry
                        readonly property bool expanded: {
                            if (!foldKey)
                                return false;
                            const storedValue = root.workspace.foldedPropertyPaths[foldKey];
                            return storedValue === undefined ? false : storedValue;
                        }

                        width: parent.width - Math.round(12 * Scaling.uiScale)
                        anchors.horizontalCenter: parent.horizontalCenter
                        height: operatorBody.height

                        color: active ? ThemeColors.dark : ThemeColors.almostdark
                        opacity: active ? 1.0 : 0.55

                        radius: 6 * Scaling.uiScale

                        MouseArea {
                            id: operatorMouseArea
                            anchors.fill: parent
                            hoverEnabled: true
                            acceptedButtons: Qt.LeftButton
                            onClicked: root.workspace.selectedOperatorAdapter = modelData
                        }

                        Column {
                            id: operatorBody
                            width: parent.width

                            Item {
                                id: operatorHeader
                                width: parent.width
                                height: Math.round(28 * Scaling.uiScale)

                                Rectangle {
                                    anchors.fill: parent
                                    color: ThemeColors.middark
                                    topLeftRadius: operatorContainer.radius
                                    topRightRadius: operatorContainer.radius
                                    bottomLeftRadius: operatorContainer.expanded ? 0 : operatorContainer.radius
                                    bottomRightRadius: operatorContainer.expanded ? 0 : operatorContainer.radius
                                }

                                Item {
                                    id: foldButton

                                    anchors {
                                        left: parent.left
                                        top: parent.top
                                        bottom: parent.bottom
                                    }
                                    width: parent.height

                                    Image {
                                        anchors.centerIn: parent
                                        width: Math.round(12 * Scaling.uiScale)
                                        fillMode: Image.PreserveAspectFit
                                        source: operatorContainer.expanded ? FontAwesome.icon("solid/caret-down") : FontAwesome.icon("solid/caret-right")
                                        opacity: foldMouseArea.containsMouse ? 1.0 : 0.75
                                    }

                                    MouseArea {
                                        id: foldMouseArea
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        onClicked: {
                                            if (operatorContainer.foldKey)
                                                root.workspace.setFoldedProperty(operatorContainer.foldKey, !operatorContainer.expanded);
                                        }
                                    }
                                }

                                // double-click to set the operator label
                                Item {
                                    id: operatorHeaderLabel

                                    property bool editing: false
                                    property string startText: ""

                                    function beginEdit() {
                                        startText = nameField.text;
                                        editing = true;
                                        nameField.forceActiveFocus();
                                        nameField.selectAll();
                                    }

                                    function commit() {
                                        if (!editing)
                                            return;
                                        const newText = nameField.text;
                                        editing = false;
                                        if (newText !== startText && operatorContainer.configAdapter)
                                            operatorContainer.configAdapter.set("label", newText);
                                    }

                                    anchors {
                                        left: foldButton.right
                                        right: rightControls.left
                                        rightMargin: Math.round(6 * Scaling.uiScale)
                                        top: parent.top
                                        bottom: parent.bottom
                                    }

                                    TextInput {
                                        id: nameField
                                        anchors.fill: parent
                                        verticalAlignment: TextEdit.AlignVCenter
                                        font: Scaling.uiFont
                                        color: operatorContainer.active ? ThemeColors.text : ThemeColors.placeholderText
                                        selectionColor: ThemeColors.highlight
                                        selectedTextColor: ThemeColors.highlightedText
                                        clip: true

                                        enabled: operatorHeaderLabel.editing
                                        selectByMouse: operatorHeaderLabel.editing

                                        onEditingFinished: operatorHeaderLabel.commit()
                                        onActiveFocusChanged: if (!activeFocus && operatorHeaderLabel.editing)
                                            operatorHeaderLabel.commit()
                                    }

                                    Binding {
                                        target: nameField
                                        property: "text"
                                        value: operatorContainer.displayName
                                        when: !operatorHeaderLabel.editing
                                        restoreMode: Binding.RestoreNone
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        enabled: !operatorHeaderLabel.editing
                                        propagateComposedEvents: true

                                        // a single click still reaches the card
                                        // underneath and selects the operator
                                        onClicked: mouse => mouse.accepted = false
                                        onDoubleClicked: operatorHeaderLabel.beginEdit()
                                    }
                                }

                                Row {
                                    id: rightControls
                                    anchors {
                                        right: parent.right
                                        rightMargin: Math.round(6 * Scaling.uiScale)
                                        verticalCenter: parent.verticalCenter
                                    }
                                    spacing: Math.round(4 * Scaling.uiScale)

                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: (operatorDelegate.index + 1) + "/" + operatorRepeater.count
                                        font: Scaling.uiSmallFont
                                        color: ThemeColors.placeholderText
                                    }

                                    CheckBox {
                                        id: operatorActiveToggle
                                        anchors.verticalCenter: parent.verticalCenter
                                        checked: operatorContainer.active
                                        onToggled: {
                                            if (operatorContainer.configAdapter)
                                                operatorContainer.configAdapter.set("active", checked);
                                        }
                                        width: Math.round(16 * Scaling.uiScale)
                                        height: Math.round(24 * Scaling.uiScale)

                                        indicator: Image {
                                            anchors.centerIn: parent
                                            source: operatorActiveToggle.checked ? FontAwesome.icon("solid/toggle-on") : FontAwesome.icon("solid/toggle-off")
                                            sourceSize: Qt.size(Math.round(9.5 * Scaling.uiScale), Math.round(9.5 * Scaling.uiScale))
                                            opacity: operatorActiveToggle.hovered ? 0.7 : 0.5
                                        }

                                        InfoToolTip {
                                            textValue: operatorContainer.active ? "Disable operator" : "Enable operator"
                                        }
                                    }

                                    IconButton {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconSource: FontAwesome.icon("solid/trash-can")
                                        iconSize: Math.round(10 * Scaling.uiScale)
                                        topPadding: Math.round(2 * Scaling.uiScale)
                                        bottomPadding: Math.round(2 * Scaling.uiScale)
                                        leftPadding: Math.round(4 * Scaling.uiScale)
                                        rightPadding: Math.round(4 * Scaling.uiScale)
                                        tooltip: "Remove operator"

                                        onClicked: root.removeOperatorRequested(operatorDelegate.index)
                                    }
                                }
                            }

                            Column {
                                visible: operatorContainer.expanded && operatorContainer.configAdapter
                                width: parent.width
                                bottomPadding: Math.round(4 * Scaling.uiScale)

                                Item {
                                    // id: operator variant header
                                    width: parent.width
                                    height: Math.round(22 * Scaling.uiScale)

                                    Text {
                                        anchors {
                                            left: parent.left
                                            leftMargin: operatorConfigEditor.fieldIndent
                                            right: parent.right
                                            rightMargin: operatorConfigEditor.fieldIndent
                                            top: parent.top
                                            topMargin: operatorConfigEditor.fieldIndent * 0.8
                                        }
                                        text: operatorContainer.configAdapter ? operatorContainer.configAdapter.displayName() : ""
                                        font: Scaling.headerSmallFont
                                        color: ThemeColors.readOnlyText
                                        elide: Text.ElideRight
                                    }
                                }

                                ConfigurationEditor {
                                    id: operatorConfigEditor
                                    configAdapter: operatorContainer.configAdapter
                                    workspace: root.workspace
                                    flattenFields: true
                                    showTypeHeader: false
                                    width: parent.width - 1
                                    anchors.left: parent.left
                                }
                            }
                        }

                        Rectangle {
                            anchors.fill: parent
                            color: "transparent"
                            radius: operatorContainer.radius
                            border.color: operatorContainer.selected ? ThemeColors.highlight : operatorMouseArea.containsMouse ? ThemeColors.midlight : ThemeColors.mid
                            border.width: Math.max(1, Math.round(1 * Scaling.uiScale))
                        }
                    }
                }
            }

            Item {
                width: parent.width
                height: Math.round(16 * Scaling.uiScale)

                Rectangle {
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: Math.max(1, Math.round(2 * Scaling.uiScale))
                    height: parent.height
                    color: ThemeColors.accent
                    opacity: 0.4
                }

                Rectangle {
                    anchors.centerIn: parent
                    width: Math.round(8 * Scaling.uiScale)
                    height: width
                    radius: width / 2
                    color: ThemeColors.accent
                    border.width: Math.max(1, Math.round(2 * Scaling.uiScale))
                    border.color: ThemeColors.dark
                }
            }

            Item {
                id: addOperator
                width: parent.width
                height: addOperatorButton.height + Math.round(4 * Scaling.uiScale)

                IconButton {
                    id: addOperatorButton
                    anchors.horizontalCenter: parent.horizontalCenter

                    text: "Add Operator"
                    iconSource: FontAwesome.icon("solid/plus")

                    onClicked: addOperatorMenu.open()

                    Menu {
                        id: addOperatorMenu

                        Repeater {
                            model: root.workspace ? root.workspace.availableOperators : []

                            MenuItem {
                                required property var modelData
                                text: modelData.label
                                onTriggered: root.addOperatorRequested(modelData.name)
                            }
                        }
                    }
                }
            }

            Item {
                id: outputLabel
                width: parent.width
                height: Math.round(18 * Scaling.uiScale)

                Row {
                    anchors.centerIn: parent
                    spacing: Math.round(4 * Scaling.uiScale)

                    Rectangle {
                        width: Math.round(6 * Scaling.uiScale)
                        height: width
                        radius: width / 2
                        color: root.operatorCount > 0 ? ThemeColors.yellow : ThemeColors.mid
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    Text {
                        text: "OUTPUT"
                        font: Scaling.uiSmallFont
                        color: ThemeColors.placeholderText
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }
        }
    }
}