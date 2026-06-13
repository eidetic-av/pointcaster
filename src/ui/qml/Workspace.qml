pragma Singleton

import QtQuick

import Pointcaster 1.0

QtObject {
    property int labelColumnWidth: 150
    property int deviceListHeight: 160
    property int streamChannelListHeight: 90

    property int selectedSessionIndex: 0

    property real inputDragSpeed: 0.5

    property list<SessionView> sessionViews: []

    readonly property var persistentPropertyNames: [
        "labelColumnWidth", "deviceListHeight", "streamChannelListHeight",
        "selectedSessionIndex"
    ]

    function applyPersistentState(state) {
        for (var key in state) {
            if (persistentPropertyNames.indexOf(key) !== -1) {
                this[key] = state[key];
            }
        }
    }

    function addSessionView(sessionView) {
        for (var i = 0; i < sessionViews.length; i++) {
            if (sessionViews[i] === sessionView)
                return;
        }
        sessionViews.push(sessionView);
    }

    function eraseSessionView(sessionView) {
        var remainingSessionViews = [];
        for (var i = 0; i < sessionViews.length; i++) {
            if (sessionViews[i] !== this)
                remainingSessionViews.push(sessionViews[i]);
        }
        sessionViews = remainingSessionViews;
    }
}
