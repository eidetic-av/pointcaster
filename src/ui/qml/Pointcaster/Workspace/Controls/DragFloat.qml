import QtQuick
import QtQuick.Controls

DoubleSpinBox {
    id: root

    property real boundValue: 0.0
    property bool boundEnabled: true
    signal commitValue(real value)

    property var minValue: undefined
    property var maxValue: undefined
    property var defaultValue: undefined

    property color focusBorderColor: DarkPalette.highlight
    property color unfocusBorderColor: "transparent"
    property color backgroundColor: (root.activeFocus || hover.hovered) ? DarkPalette.almostdark : "transparent"

    property real dragThresholdPx: 6.0
    property real pixelsPerStep: 6.0
    property real accelerateAfterPx: 120.0

    editable: true
    enabled: boundEnabled

    readonly property real effectiveFrom: {
        var n = Number(minValue);
        return (minValue === undefined || minValue === null || isNaN(n)) ? -1.0e308 : n;
    }
    readonly property real effectiveTo: {
        var n = Number(maxValue);
        return (maxValue === undefined || maxValue === null || isNaN(n)) ?  1.0e308 : n;
    }

    from: effectiveFrom
    to: effectiveTo

    function clampToRange(v) {
        var asNumber = Number(v);
        if (isNaN(asNumber))
            asNumber = 0.0;
        return Math.max(from, Math.min(to, asNumber));
    }

    readonly property real boundedRange: {
        var r = root.to - root.from;
        return (Number.isFinite(r) && r > 0.0 && r < 1.0e307) ? r : NaN;
    }

    readonly property real dragTargetPixels: {
        var frac = Math.max(0.05, Math.min(1.0, WorkspaceState.inputDragSpeed));
        var w = 15000.0;
        return Math.max(200.0, w * frac);
    }

    readonly property real dragStepSize: {
        if (!Number.isFinite(root.boundedRange))
            return root.stepSize;

        var valuePerPixel = root.boundedRange / root.dragTargetPixels;
        var step = valuePerPixel * root.pixelsPerStep;
        return Math.max(step, 1e-12);
    }

    function hasValidDefault() {
        var n = Number(defaultValue);
        return !(defaultValue === undefined || defaultValue === null || isNaN(n));
    }

    function resetToDefault() {
        if (!hasValidDefault())
            return;
        var v = clampToRange(Number(defaultValue));
        if (v === value)
            return;
        value = v;
    }

    Component.onCompleted: value = clampToRange(boundValue);

    onValueModified: {
        boundValue = value;
        commitValue(value);
    }

    onActiveFocusChanged: {
        if (!activeFocus) {
            boundValue = value;
            commitValue(value);
        }
    }

    HoverHandler {
        id: hover
        acceptedDevices: PointerDevice.Mouse
    }

    background: Rectangle {
        color: root.backgroundColor
        border.color: root.activeFocus ? root.focusBorderColor : root.unfocusBorderColor
        border.width: 1
        radius: 0
    }

    contentItem: TextInput {
        id: spinTextInput
        z: 2
        text: root.displayText

        font: root.font
        color: DarkPalette.text
        selectionColor: DarkPalette.highlight
        selectedTextColor: DarkPalette.highlightedText
        horizontalAlignment: Qt.AlignHLeft
        verticalAlignment: Qt.AlignVCenter

        readOnly: !root.editable
        validator: root.validator
        inputMethodHints: root.inputMethodHints
        clip: false

        onActiveFocusChanged: {
            if (activeFocus && !dragBehaviour.dragModeActive)
                Qt.callLater(function () { spinTextInput.selectAll(); });
        }

        DragInput {
            id: dragBehaviour
            anchors.fill: parent

            targetTextInput: spinTextInput
            focusTarget: root

            enabled: root.enabled
            editable: root.editable

            dragThresholdPx: root.dragThresholdPx
            pixelsPerStep: root.pixelsPerStep
            accelerateAfterPx: root.accelerateAfterPx

            quantiseDeltaToInteger: false

            numericValue: function () { return root.value; }
            setNumericValue: function (v) {
                var clamped = root.clampToRange(v);
                if (clamped !== root.value)
                    root.value = clamped;
            }

            stepSize: function () { return root.dragStepSize; }

            modifierMultiplier: function (mods) {
                if (mods & Qt.ShiftModifier)   return 10.0;
                if (mods & Qt.ControlModifier) return 0.1;
                return 1.0;
            }

            resetToDefault: function () { root.resetToDefault(); }

            onEdited: {
                root.boundValue = root.value;
                root.commitValue(root.value);
            }
            onCommitted: {
                root.boundValue = root.value;
                root.commitValue(root.value);
            }
        }
    }

    onBoundValueChanged: {
        if (spinTextInput.activeFocus || dragBehaviour.dragModeActive)
            return;

        var clamped = clampToRange(boundValue);
        if (root.value !== clamped)
            root.value = clamped;
    }

    onEffectiveFromChanged: {
        var clamped = clampToRange(root.value);
        if (clamped !== root.value) {
            root.value = clamped;
            boundValue = clamped;
            commitValue(clamped);
        }
    }

    onEffectiveToChanged: {
        var clamped = clampToRange(root.value);
        if (clamped !== root.value) {
            root.value = clamped;
            boundValue = clamped;
            commitValue(clamped);
        }
    }
}
