// Copyright (C) 2022 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR GPL-3.0-only

import QtQuick
import QtQuick.Window
import QtQuick3D

import Pointcaster 1.0

Item {
    id: root
    required property Node origin
    required property Camera camera

    property real xSpeed: 0.16666666667
    property real ySpeed: 0.83333333333
    property real panScale: 0.002

    property bool xInvert: false
    property bool yInvert: true

    property bool mouseEnabled: true
    property bool panEnabled: true

    implicitWidth: parent.width
    implicitHeight: parent.height

    readonly property bool inputsNeedProcessing: viewState.useMouse || viewState.isPanning

    property alias mouseHeld: viewState.mouseHeld
    property alias scrolling: viewState.isScrolling
    property alias dragActive: rmbDragHandler.active

    property real shaderPointSize: AppSettings.pointSize * 0.1
    property real shaderViewportHeight: height * Screen.devicePixelRatio

    DragHandler {
        id: dragHandler
        target: null
        enabled: root.mouseEnabled

        acceptedButtons: Qt.LeftButton
        acceptedModifiers: Qt.NoModifier

        onCentroidChanged: {
            root.mouseMoved(Qt.vector2d(centroid.position.x, centroid.position.y), false);
        }

        onActiveChanged: {
            if (active)
                root.mousePressed(Qt.vector2d(centroid.position.x, centroid.position.y));
            else
                root.mouseReleased(Qt.vector2d(centroid.position.x, centroid.position.y));
        }
    }

    DragHandler {
        id: rmbDragHandler
        target: null
        enabled: root.mouseEnabled && root.panEnabled

        acceptedButtons: Qt.RightButton
        acceptedModifiers: Qt.NoModifier

        property real lastX: 0
        property real lastY: 0

        onActiveChanged: {
            if (active) {
                lastX = translation.x;
                lastY = translation.y;
            }
        }
        onTranslationChanged: {
            const dx = translation.x - lastX;
            const dy = translation.y - lastY;
            lastX = translation.x;
            lastY = translation.y;
            const panScale = Math.abs(root.camera.z) * root.panScale;
            const sceneDelta = Qt.vector3d(root.camera.right.x, root.camera.right.y, root.camera.right.z).times(-dx * panScale).plus(Qt.vector3d(root.camera.up.x, root.camera.up.y, root.camera.up.z).times(dy * panScale));
            const parentNode = origin.parent;
            const parentDelta = parentNode && parentNode.mapDirectionFromScene ? parentNode.mapDirectionFromScene(sceneDelta) : sceneDelta;
            origin.position = Qt.vector3d(origin.position.x + parentDelta.x, origin.position.y + parentDelta.y, origin.position.z + parentDelta.z);
        }
    }

    WheelHandler {
        id: wheelHandler
        orientation: Qt.Vertical
        target: null
        enabled: root.mouseEnabled
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
        onWheel: event => {
            let delta = -event.angleDelta.y * 0.01;
            root.camera.z += root.camera.z * 0.1 * delta;
        }
        onActiveChanged: {
            viewState.isScrolling = wheelHandler.active;
        }
    }

    function mousePressed(newPos) {
        root.forceActiveFocus();
        viewState.currentPos = newPos;
        viewState.lastPos = newPos;
        viewState.useMouse = true;
        viewState.mouseHeld = true;
    }

    function mouseReleased(newPos) {
        viewState.useMouse = false;
        viewState.mouseHeld = false;
    }

    function mouseMoved(newPos: vector2d) {
        viewState.currentPos = newPos;
    }

    function startPan(pos: vector2d) {
        viewState.isPanning = true;
        viewState.currentPanPos = pos;
        viewState.lastPanPos = pos;
    }

    function endPan() {
        viewState.isPanning = false;
    }

    function panEvent(newPos: vector2d) {
        viewState.currentPanPos = newPos;
    }

    FrameAnimation {
        id: updateTimer
        running: root.inputsNeedProcessing
        onTriggered: viewState.processInput(frameTime * 100)
    }

    QtObject {
        id: viewState

        property bool useMouse: false
        property bool mouseHeld: false
        property bool isScrolling: false
        property bool isPanning: false

        property vector2d lastPos: Qt.vector2d(0, 0)
        property vector2d lastPanPos: Qt.vector2d(0, 0)
        property vector2d currentPos: Qt.vector2d(0, 0)
        property vector2d currentPanPos: Qt.vector2d(0, 0)

        function negate(vector) {
            return Qt.vector3d(-vector.x, -vector.y, -vector.z);
        }

        function processInput(frameDelta) {
            if (useMouse) {
                // Get the delta
                var rotationVector = root.origin.eulerRotation;
                var delta = Qt.vector2d(lastPos.x - currentPos.x, lastPos.y - currentPos.y);
                // rotate x
                var rotateX = delta.x * root.xSpeed;
                if (root.xInvert)
                    rotateX = -rotateX;
                rotationVector.y += rotateX;

                // rotate y
                var rotateY = delta.y * -root.ySpeed;
                if (root.yInvert)
                    rotateY = -rotateY;
                rotationVector.x += rotateY;
                root.origin.setEulerRotation(rotationVector);
                lastPos = currentPos;
            }
            if (isPanning) {
                let delta = currentPanPos.minus(lastPanPos);
                delta.x = -delta.x;

                delta.x = (delta.x / root.width) * root.camera.z;
                delta.y = (delta.y / root.height) * root.camera.z;

                let velocity = Qt.vector3d(0, 0, 0);
                // X Movement
                let xDirection = root.origin.right;
                velocity = velocity.plus(Qt.vector3d(xDirection.x * delta.x, xDirection.y * delta.x, xDirection.z * delta.x));
                // Y Movement
                let yDirection = root.origin.up;
                velocity = velocity.plus(Qt.vector3d(yDirection.x * delta.y, yDirection.y * delta.y, yDirection.z * delta.y));

                root.origin.position = root.origin.position.plus(velocity);

                lastPanPos = currentPanPos;
            }
        }
    }
}
