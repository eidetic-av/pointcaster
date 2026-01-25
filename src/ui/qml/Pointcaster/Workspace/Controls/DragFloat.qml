import QtQuick
import QtQuick.Controls

DoubleSpinBox {
    id: root

    property real boundValue: 0.0
    signal commitValue(real value)

    property var minValue: undefined
    property var maxValue: undefined
    property var defaultValue: undefined

    property color focusBorderColor: DarkPalette.highlight
    property color unfocusBorderColor: "transparent"
    property color backgroundColor: (root.activeFocus || hover.hovered) ? DarkPalette.almostdark : "transparent"

    editable: true
    decimals: width >= 58 ? 3 : 2

    readonly property bool showArrowButtons: width >= 48
    rightPadding: showArrowButtons ? 12 : 0
    up.indicator.visible: showArrowButtons
    down.indicator.visible: showArrowButtons
    up.indicator.enabled: showArrowButtons
    down.indicator.enabled: showArrowButtons

    property real dragThresholdPx: 6.0
    property real pixelsPerStep: 6.0
    property real accelerateAfterPx: 120.0

    readonly property real defaultFromValue: -10.0
    readonly property real defaultToValue: 10.0
    readonly property real defaultResetValue: 0.0

    readonly property bool hasMin: {
        var n = Number(minValue);
        return !(minValue === undefined || minValue === null || isNaN(n));
    }
    readonly property bool hasMax: {
        var n = Number(maxValue);
        return !(maxValue === undefined || maxValue === null || isNaN(n));
    }
    readonly property bool hasDefault: {
        var n = Number(defaultValue);
        return !(defaultValue === undefined || defaultValue === null || isNaN(n));
    }

    // only clamp when both bounds are specified
    readonly property bool clampEnabled: hasMin && hasMax

    readonly property real uiFrom: hasMin ? Number(minValue) : defaultFromValue
    readonly property real uiTo: hasMax ? Number(maxValue) : defaultToValue
    readonly property real effectiveDefault: hasDefault ? Number(defaultValue) : defaultResetValue

    // what the spinbox itself clamps to...
    readonly property real effectiveFrom: clampEnabled ? Number(minValue) : -1e308
    readonly property real effectiveTo: clampEnabled ? Number(maxValue) : 1e308

    from: effectiveFrom
    to: effectiveTo

    function clampToRange(v) {
        var x = Number(v);
        if (isNaN(x))
            x = defaultResetValue;

        if (!clampEnabled)
            return x;

        return Math.max(root.from, Math.min(root.to, x));
    }

    readonly property real boundedRange: {
        if (!clampEnabled)
            return 5;

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

    function resetToDefault() {
        var v = clampToRange(effectiveDefault);
        if (v === value)
            return;
        value = v;
    }

    Component.onCompleted: value = clampToRange(boundValue)

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

            // forward font to DragInput so its edit menu matches
            font: root.font

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

    onClampEnabledChanged: {
        var clamped = clampToRange(root.value);
        if (clamped !== root.value) {
            root.value = clamped;
            boundValue = clamped;
            commitValue(clamped);
        }
    }
    onMinValueChanged: {
        if (!clampEnabled)
            return;
        var clamped = clampToRange(root.value);
        if (clamped !== root.value) {
            root.value = clamped;
            boundValue = clamped;
            commitValue(clamped);
        }
    }
    onMaxValueChanged: {
        if (!clampEnabled)
            return;
        var clamped = clampToRange(root.value);
        if (clamped !== root.value) {
            root.value = clamped;
            boundValue = clamped;
            commitValue(clamped);
        }
    }
}
