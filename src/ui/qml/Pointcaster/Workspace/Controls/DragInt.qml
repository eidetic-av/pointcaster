import QtQuick
import QtQuick.Controls

SpinBox {
    id: root

    property int boundValue: 0
    property bool boundEnabled: true
    signal commitValue(int value)

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
    stepSize: 1

    readonly property int effectiveFrom: {
        var n = Number(minValue);
        return (minValue === undefined || minValue === null || isNaN(n)) ? -2147483648 : Math.trunc(n);
    }
    readonly property int effectiveTo: {
        var n = Number(maxValue);
        return (minValue === undefined || minValue === null || isNaN(n)) ?  2147483647 : Math.trunc(n);
    }

    from: effectiveFrom
    to: effectiveTo

    function clampToRange(v) {
        var n = Number(v);
        if (isNaN(n))
            n = 0;
        return Math.max(from, Math.min(to, Math.trunc(n)));
    }

    readonly property real boundedRange: {
        var r = Number(root.to) - Number(root.from);
        return (Number.isFinite(r) && r > 0 && r < 4.0e9) ? r : NaN;
    }

    readonly property real dragTargetPixels: {
        var frac = Math.max(0.05, Math.min(1.0, WorkspaceState.inputDragSpeed));
        var w = 15000.0;
        return Math.max(200.0, w * frac);
    }

    readonly property real dragStepSize: {
        if (!Number.isFinite(root.boundedRange))
            return 1.0;

        var unitsPerPixel = root.boundedRange / root.dragTargetPixels;
        var step = unitsPerPixel * root.pixelsPerStep;

        // ensure at least 1 unit step after quantisation for reasonable drags
        return Math.max(step, 1e-6);
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
        text: root.textFromValue(root.value, root.locale)

        font: root.font
        color: DarkPalette.text
        selectionColor: DarkPalette.highlight
        selectedTextColor: DarkPalette.highlightedText
        horizontalAlignment: Qt.AlignHLeft
        verticalAlignment: Qt.AlignVCenter

        readOnly: !root.editable
        validator: root.validator
        inputMethodHints: Qt.ImhFormattedNumbersOnly
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

            quantiseDeltaToInteger: true

            numericValue: function () { return root.value; }
            setNumericValue: function (v) {
                var clamped = root.clampToRange(v);
                if (clamped !== root.value)
                    root.value = clamped;
            }

            stepSize: function () { return root.dragStepSize; }

            resetToDefault: function () { root.resetToDefault(); }

            modifierMultiplier: function (mods) {
                if (mods & Qt.ShiftModifier)   return 10.0;
                if (mods & Qt.ControlModifier) return 0.1;
                return 1.0;
            }

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
