import QtQuick
import QtQuick.Controls

import Pointcaster 1.0
import Pointcaster.Workspace 1.0

Item {
    id: root

    property var model: null
    property string emptyTitle: "Properties"

    // ---- Layout
    property int labelColumnWidth: WorkspaceState.labelColumnWidth
    property int minLabelColumnWidth: Math.round(40 * Scaling.uiScale)
    property int minValueColumnWidth: Math.round(170 * Scaling.uiScale)
    property bool expanded: true
    property int outerHorizontalMargin: Math.round(8 * Scaling.uiScale)
    property int outerTopMargin: Math.round(4 * Scaling.uiScale)

    readonly property real _viewportWidth: {
        if (ScrollView.view)
            return ScrollView.view.width;
        if (parent)
            return parent.width;
        return 0;
    }
    width: _viewportWidth

    implicitHeight: root.outerTopMargin + inspectorGroup.implicitHeight

    Rectangle {
        id: inspectorGroup
        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
            leftMargin: root.outerHorizontalMargin
            rightMargin: root.outerHorizontalMargin
            topMargin: root.outerTopMargin
        }

        implicitHeight: header.height + (root.expanded ? body.implicitHeight : 0)

        color: ThemeColors.dark
        border.color: ThemeColors.mid
        border.width: Math.max(1, Math.round(1 * Scaling.uiScale))
        clip: true

        Rectangle {
            id: header
            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
            }
            height: Math.round(28 * Scaling.uiScale)
            color: headerMouseArea.containsMouse ? ThemeColors.mid : (root.expanded ? ThemeColors.middark : ThemeColors.base)
            border.color: ThemeColors.mid
            border.width: Math.max(1, Math.round(1 * Scaling.uiScale))

            Image {
                id: headerArrowIcon
                width: Math.round(12 * Scaling.uiScale)
                height: Math.round(12 * Scaling.uiScale)
                fillMode: Image.PreserveAspectFit
                source: root.expanded ? FontAwesome.icon("solid/caret-down") : FontAwesome.icon("solid/caret-right")
                opacity: 0.75
                anchors {
                    left: parent.left
                    leftMargin: Math.round(3 * Scaling.uiScale)
                    verticalCenter: parent.verticalCenter
                }
            }

            Text {
                id: headerText
                anchors {
                    left: headerArrowIcon.left
                    right: parent.right
                    leftMargin: Math.round(15 * Scaling.uiScale)
                    rightMargin: Math.round(10 * Scaling.uiScale)
                    verticalCenter: parent.verticalCenter
                }
                text: root.model ? (root.model.displayName() + " Properties") : root.emptyTitle
                elide: Text.ElideRight
                font: Scaling.uiFont
                color: ThemeColors.text
            }

            MouseArea {
                id: headerMouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: root.expanded = !root.expanded
            }
        }

        Item {
            id: body
            anchors {
                top: header.bottom
                left: parent.left
                right: parent.right
            }
            visible: root.expanded

            implicitHeight: root.model ? fieldsContainer.implicitHeight : (noSelectionText.implicitHeight + Math.round(8 * Scaling.uiScale))

            Text {
                id: noSelectionText
                anchors {
                    left: parent.left
                    right: parent.right
                    top: parent.top
                    leftMargin: Math.round(10 * Scaling.uiScale)
                    rightMargin: Math.round(10 * Scaling.uiScale)
                    topMargin: Math.round(3 * Scaling.uiScale)
                }
                visible: !root.model
                text: "No device selected"
                font: Scaling.uiFont
                color: ThemeColors.text
            }

            Item {
                id: fieldsArea
                anchors {
                    top: parent.top
                    left: parent.left
                    right: parent.right
                    rightMargin: Math.max(1, Math.round(1 * Scaling.uiScale))
                }
                visible: root.model
                height: fieldsContainer.implicitHeight

                readonly property int _labelLeftMargin: Math.round(6 * Scaling.uiScale)
                readonly property int _valueLeftOverlap: Math.round(5 * Scaling.uiScale)

                readonly property real availableInnerWidth: Math.max(0, width - _labelLeftMargin + _valueLeftOverlap)

                readonly property int effectiveLabelWidth: {
                    var preferred = root.labelColumnWidth;

                    var maxLabel = Math.floor(availableInnerWidth - root.minValueColumnWidth);
                    if (!Number.isFinite(maxLabel))
                        maxLabel = 0;

                    var w = Math.max(root.minLabelColumnWidth, preferred);
                    w = Math.min(w, maxLabel);
                    return Math.max(0, w);
                }

                readonly property int dragMinLabel: root.minLabelColumnWidth
                readonly property int dragMaxLabel: Math.max(root.minLabelColumnWidth, Math.floor(availableInnerWidth - root.minValueColumnWidth))

                Column {
                    id: fieldsContainer
                    width: fieldsArea.width
                    spacing: 0

                    Repeater {
                        model: root.model ? root.model.fieldCount() : 0

                        delegate: Item {
                            required property int index

                            width: fieldsContainer.width
                            height: Math.max(nameText.implicitHeight, valueContainer.implicitHeight) + Math.round(6 * Scaling.uiScale)
                            visible: root.model && !root.model.isHidden(index)

                            Item {
                                id: labelContainer
                                anchors {
                                    top: parent.top
                                    bottom: parent.bottom
                                    left: parent.left
                                    leftMargin: fieldsArea._labelLeftMargin
                                }
                                width: fieldsArea.effectiveLabelWidth

                                Text {
                                    id: nameText
                                    anchors {
                                        left: parent.left
                                        right: parent.right
                                        rightMargin: Math.round(10 * Scaling.uiScale)
                                        verticalCenter: parent.verticalCenter
                                    }
                                    horizontalAlignment: Text.AlignLeft
                                    color: ThemeColors.text
                                    text: root.model ? StringUtils.titleFromSnake(root.model.fieldName(index)) : ""
                                    font: Scaling.uiFont
                                    elide: Text.ElideRight
                                }

                                InfoToolTip {
                                    visible: labelHover.hovered
                                    textValue: root.model ? "osc/address/" + qsTr(root.model.fieldName(index)) : ""
                                    x: Math.max((labelContainer.width - implicitWidth) * 0.5, Math.round(4 * Scaling.uiScale))
                                    y: labelContainer.height + Math.round(4 * Scaling.uiScale)
                                }

                                HoverHandler {
                                    id: labelHover
                                }
                            }

                            Loader {
                                id: valueContainer
                                anchors {
                                    left: labelContainer.right
                                    leftMargin: -fieldsArea._valueLeftOverlap
                                    right: parent.right
                                    verticalCenter: parent.verticalCenter
                                }
                                z: 3

                                readonly property string typeName: root.model ? String(root.model.fieldTypeName(index)).toLowerCase() : ""

                                sourceComponent: {
                                    if (root.model && root.model.isEnum(index))
                                        return enumEditor;
                                    if (typeName === "int" || typeName === "int32" || typeName === "int32_t" || typeName === "integer")
                                        return intEditor;
                                    if (typeName === "float" || typeName === "float32" || typeName === "float32_t" || typeName === "double" || typeName === "real" || typeName === "number")
                                        return floatEditor;
                                    if (typeName === "pc::float3")
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

                                    text: root.model ? String(root.model.fieldValue(index)) : ""
                                    enabled: root.model ? !root.model.isDisabled(index) : false
                                    opacity: enabled ? 1.0 : 0.66

                                    onEditingFinished: {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, text);
                                    }

                                    Connections {
                                        target: root.model
                                        function onFieldChanged(changedIndex) {
                                            if (changedIndex !== index)
                                                return;
                                            valueField.text = root.model ? String(root.model.fieldValue(index)) : "";
                                        }
                                    }
                                }
                            }

                            Component {
                                id: intEditor
                                DragInt {
                                    id: intEditorControl
                                    font: Scaling.uiFont

                                    minValue: root.model ? root.model.minMax(index)[0] : undefined
                                    maxValue: root.model ? root.model.minMax(index)[1] : undefined
                                    defaultValue: root.model ? root.model.defaultValue(index) : undefined

                                    boundValue: {
                                        if (!root.model)
                                            return 0;
                                        var n = Number(root.model.fieldValue(index));
                                        return isNaN(n) ? 0 : Math.trunc(n);
                                    }

                                    onCommitValue: function (v) {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, v);
                                    }

                                    Connections {
                                        target: root.model
                                        function onFieldChanged(changedIndex) {
                                            if (changedIndex !== index)
                                                return;
                                            var n = Number(root.model.fieldValue(index));
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

                                    minValue: root.model ? root.model.minMax(index)[0] : undefined
                                    maxValue: root.model ? root.model.minMax(index)[1] : undefined
                                    defaultValue: root.model ? root.model.defaultValue(index) : undefined

                                    boundValue: {
                                        if (!root.model)
                                            return 0.0;
                                        var n = Number(root.model.fieldValue(index));
                                        return isNaN(n) ? 0.0 : n;
                                    }

                                    onCommitValue: function (v) {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, v);
                                    }

                                    Connections {
                                        target: root.model
                                        function onFieldChanged(changedIndex) {
                                            if (changedIndex !== index)
                                                return;
                                            var n = Number(root.model.fieldValue(index));
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

                                    property var minMax: root.model ? root.model.minMax(index) : undefined

                                    minValue: minMax ? minMax[0] : undefined
                                    maxValue: minMax ? minMax[1] : undefined

                                    boundValue: {
                                        if (!root.model)
                                            return Qt.vector3d(0, 0, 0);
                                        const v = root.model.fieldValue(index); // expected QVector3D
                                        return Qt.vector3d(Number(v.x) || 0, Number(v.y) || 0, Number(v.z) || 0);
                                    }

                                    onCommitValue: function (v3) {
                                        if (!root.model)
                                            return;
                                        root.model.setFloat3Field(index, v3.x, v3.y, v3.z);
                                    }

                                    Connections {
                                        target: root.model
                                        function onFieldChanged(changedIndex) {
                                            if (changedIndex !== index)
                                                return;
                                            const v = root.model.fieldValue(index);
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

                                    enabled: root.model ? !root.model.isDisabled(index) : false
                                    opacity: enabled ? 1.0 : 0.66

                                    options: root.model ? root.model.enumOptions(index) : []
                                    boundValue: {
                                        if (!root.model)
                                            return 0;
                                        var n = Number(root.model.fieldValue(index));
                                        return isNaN(n) ? 0 : Math.trunc(n);
                                    }

                                    onCommitValue: function (v) {
                                        if (!root.model)
                                            return;
                                        root.model.setFieldValue(index, v);
                                    }

                                    Connections {
                                        target: root.model
                                        function onFieldChanged(changedIndex) {
                                            if (changedIndex !== index)
                                                return;
                                            var n = Number(root.model.fieldValue(index));
                                            enumSelector.boundValue = isNaN(n) ? 0 : Math.trunc(n);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    id: dividerHandle
                    x: (fieldsArea.effectiveLabelWidth - width / 2) + Math.max(1, Math.round(1 * Scaling.uiScale))
                    z: 99
                    width: Math.round(5 * Scaling.uiScale)
                    anchors {
                        top: parent.top
                        bottom: parent.bottom
                        bottomMargin: Math.max(1, Math.round(1 * Scaling.uiScale))
                    }
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
                            var dividerCentreX = dividerHandle.x + mouseX;
                            var desiredLabelWidth = dividerCentreX - fieldsArea._labelLeftMargin;
                            var clamped = Math.max(fieldsArea.dragMinLabel, Math.min(fieldsArea.dragMaxLabel, desiredLabelWidth));
                            WorkspaceState.labelColumnWidth = Math.round(clamped);
                        }
                    }
                }
            }
        }
    }
}
