import QtQuick
import QtQuick.Controls

Item {
    id: root
    focus: false

    required property TextInput targetTextInput
    property Item focusTarget: null

    property bool enabled: true
    property bool editable: true

    property font font: Qt.font({ pointSize: 12 })

    // drag tuning
    property real dragThresholdPx: 6.0
    property real pixelsPerStep: 6.0
    property real accelerateAfterPx: 120.0

    property var numericValue: null
    property var setNumericValue: null
    property var stepSize: null

    property var modifierMultiplier: null
    property var resetToDefault: null

    property bool quantiseDeltaToInteger: false

    property real deltaEpsilon: 0.0

    signal edited()
    signal committed()

    property bool dragModeActive: false

    property real pressX: 0.0
    property real pressValue: 0.0
    property real lastAppliedDeltaValue: 0.0
    property int lastMods: 0

    function _forceFocus() {
        if (focusTarget)
            focusTarget.forceActiveFocus();
        else
            root.forceActiveFocus();
    }

    function _defaultMultiplier(mods) {
        if (mods & Qt.ControlModifier) return 0.1;
        if (mods & Qt.ShiftModifier)   return 10.0;
        return 1.0;
    }

    function _multiplier(mods) {
        return modifierMultiplier ? Number(modifierMultiplier(mods)) : _defaultMultiplier(mods);
    }

    function _stepSize() {
        var s = stepSize ? Number(stepSize()) : 1.0;
        if (!Number.isFinite(s) || s === 0.0) return 1.0;
        return s;
    }

    function _effectiveDeltaEpsilon(mods) {
        if (root.deltaEpsilon > 0.0)
            return root.deltaEpsilon;
        // 1% of a single step with current modifiers
        var e = Math.abs(root._stepSize() * root._multiplier(mods)) * 0.01;
        return Math.max(e, 1e-12);
    }

    function _quantiseDelta(deltaValue) {
        if (!root.quantiseDeltaToInteger)
            return deltaValue;
        return Math.round(deltaValue);
    }

    function _selectAllLater() {
        Qt.callLater(function () { root.targetTextInput.selectAll(); });
    }

    Menu {
        id: editMenu
        font: root.font

        Action {
            text: qsTr("Cut")
            enabled: root.enabled && root.editable && root.targetTextInput.selectedText.length > 0
            onTriggered: root.targetTextInput.cut()
        }
        Action {
            text: qsTr("Copy")
            enabled: root.targetTextInput.selectedText.length > 0
            onTriggered: root.targetTextInput.copy()
        }
        Action {
            text: qsTr("Paste")
            enabled: root.enabled && root.editable
            onTriggered: root.targetTextInput.paste()
        }
    }

    MouseArea {
        id: gestureArea
        // attach to the TextInput so coords match what user is dragging on
        parent: root.targetTextInput
        anchors.fill: parent

        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        preventStealing: true
        propagateComposedEvents: false

        onPressed: (mouse) => {
            if (!root.enabled) {
                mouse.accepted = false;
                return;
            }

            if (mouse.button === Qt.RightButton) {
                root._forceFocus();
                editMenu.popup();
                mouse.accepted = true;
                return;
            }

            if (mouse.modifiers & Qt.AltModifier) {
                root._forceFocus();
                root.dragModeActive = false;
                if (root.resetToDefault)
                    root.resetToDefault();
                mouse.accepted = true;
                return;
            }

            root.pressX = mouse.x; // TextInput-local
            root.pressValue = root.numericValue ? Number(root.numericValue()) : 0.0;
            root.lastAppliedDeltaValue = 0.0;
            root.dragModeActive = false;
            root.lastMods = mouse.modifiers;

            root._forceFocus();
            mouse.accepted = true;
        }

        onPositionChanged: (mouse) => {
            if (!gestureArea.pressed || (mouse.buttons & Qt.LeftButton) === 0)
                return;

            if (!root.numericValue || !root.setNumericValue)
                return;

            if (!root.dragModeActive) {
                if (Math.abs(mouse.x - root.pressX) >= root.dragThresholdPx) {
                    root.dragModeActive = true;
                    root.targetTextInput.deselect();
                } else {
                    return;
                }
            }

            // if modifiers change mid-drag, rebase so we don't jump.
            if (mouse.modifiers !== root.lastMods) {
                root.pressX = mouse.x;
                root.pressValue = Number(root.numericValue());
                root.lastAppliedDeltaValue = 0.0;
                root.lastMods = mouse.modifiers;
            }

            var dx = mouse.x - root.pressX;
            var baseSteps = dx / root.pixelsPerStep;

            var absDx = Math.abs(dx);
            if (absDx > root.accelerateAfterPx) {
                var accel = 1 + Math.trunc((absDx - root.accelerateAfterPx) / 80.0);
                baseSteps = baseSteps * accel;
            }

            var mult = root._multiplier(mouse.modifiers);
            var deltaValue = baseSteps * root._stepSize() * mult;
            deltaValue = root._quantiseDelta(deltaValue);

            var eps = root._effectiveDeltaEpsilon(mouse.modifiers);
            if (Math.abs(deltaValue - root.lastAppliedDeltaValue) < eps)
                return;

            if (root.quantiseDeltaToInteger && deltaValue === root.lastAppliedDeltaValue)
                return;

            root.lastAppliedDeltaValue = deltaValue;

            root.setNumericValue(root.pressValue + deltaValue);
            root.edited();
            mouse.accepted = true;
        }

        onReleased: (mouse) => {
            if (mouse.button === Qt.RightButton) {
                mouse.accepted = true;
                return;
            }

            if (!root.dragModeActive) {
                root._selectAllLater();
            } else {
                root.committed();
            }

            root.dragModeActive = false;
            mouse.accepted = true;
        }

        onCanceled: root.dragModeActive = false

        onWheel: (wheel) => {
            var hasFocus = root.focusTarget ? root.focusTarget.activeFocus : root.activeFocus;
            if (!hasFocus || !root.enabled) {
                wheel.accepted = false;
                return;
            }

            if (!root.numericValue || !root.setNumericValue) {
                wheel.accepted = false;
                return;
            }

            var notches = wheel.angleDelta.y / 120.0;
            if (notches === 0.0) {
                wheel.accepted = true;
                return;
            }

            var mult = root._multiplier(wheel.modifiers);
            var deltaValue = notches * root._stepSize() * mult;
            deltaValue = root._quantiseDelta(deltaValue);

            // in quantised mode, ignore tiny deltas that round to 0.
            if (deltaValue === 0.0) {
                wheel.accepted = true;
                return;
            }

            var current = Number(root.numericValue());
            root.setNumericValue(current + deltaValue);
            root.edited();

            wheel.accepted = true;
        }
    }
}
