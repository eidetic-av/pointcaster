pragma Singleton

import QtQuick

import Pointcaster 1.0

QtObject {
    property int labelColumnWidth: 150
    property real inputDragSpeed: 0.5

    property int selectedSessionIndex: 0

    property list<SessionView> sessionViews: []

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
