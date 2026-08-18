pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Shapes
import QtQuick3D

import Pointcaster 1.0

import Gizmo3D

// Push/pull editing of a distance out from a point
Item {
    id: root

    property View3D view3d: null

    property real radius: 0

    // parent space -> world space
    property matrix4x4 parentWorld: Qt.matrix4x4()

    property real limitLow: 0
    property real limitHigh: 1000

    property int snapModifier: Qt.ControlModifier

    property real snapIncrement: 10

    property color ringColor: ThemeColors.yellow
    property real lineWidth: 1

    readonly property var handleAngles: [0, 90, 180, 270]

    property int segments: 96

    property real puckSize: 10
    property real hitRadius: 13

    anchors.fill: parent

    // ── Live drag state ──

    property bool dragging: false
    property bool snapping: false

    property real dragRadius: 0

    readonly property real liveRadius: dragging ? dragRadius : radius

    signal radiusPreview(real newRadius)
    signal radiusCommitted(real newRadius)

    readonly property bool ready: view3d !== null

    // ── Geometry helpers ──

    function _rimPoint(degrees, distance) {
        const radians = degrees * Math.PI / 180;
        return Qt.vector3d(Math.cos(radians) * distance, 0, Math.sin(radians) * distance);
    }

    // the parent's translation drops out, leaving the direction the rim moves
    // along as the world sees it
    function _directionWorld(degrees) {
        const origin = root.parentWorld.times(Qt.vector3d(0, 0, 0));
        const along = root.parentWorld.times(root._rimPoint(degrees, 1));
        return GizmoMath.normalize(GizmoMath.vectorSubtract(along, origin));
    }

    // ── Drag plumbing ──

    property real _dragAngle: 0
    property vector3d _dragRimWorld: Qt.vector3d(0, 0, 0)
    property vector3d _dragAxisWorld: Qt.vector3d(0, 0, 0)
    property real _dragInitialParameter: 0

    function _axisParameter(screenX, screenY) {
        var ray = GizmoMath.getCameraRay(root.view3d, Qt.point(screenX, screenY));
        return -GizmoMath.closestPointOnAxisToRay(ray.origin, ray.direction, root._dragRimWorld, root._dragAxisWorld);
    }

    function _beginDrag(angle, screenX, screenY) {
        root._dragAngle = angle;
        // seeded before dragging flips, since liveRadius reads this once it does
        root.dragRadius = root.radius;
        root._dragRimWorld = root.parentWorld.times(root._rimPoint(angle, root.radius));
        root._dragAxisWorld = root._directionWorld(angle);
        root._dragInitialParameter = root._axisParameter(screenX, screenY);
        root.dragging = true;
    }

    function _updateDrag(screenX, screenY, modifiers) {
        if (!root.dragging)
            return;

        root.snapping = (modifiers & root.snapModifier) !== 0;

        var delta = root._axisParameter(screenX, screenY) - root._dragInitialParameter;
        var moved = Qt.vector3d(root._dragRimWorld.x + root._dragAxisWorld.x * delta, root._dragRimWorld.y + root._dragAxisWorld.y * delta, root._dragRimWorld.z + root._dragAxisWorld.z * delta);

        var localPoint = root.parentWorld.inverted().times(moved);
        var direction = root._rimPoint(root._dragAngle, 1);
        var value = localPoint.x * direction.x + localPoint.z * direction.z;

        if (root.snapping)
            value = GizmoMath.snapValueAbsolute(value, root.snapIncrement);

        root.dragRadius = Math.max(root.limitLow, Math.min(value, root.limitHigh));
        root.radiusPreview(root.dragRadius);
    }

    function _endDrag() {
        if (!root.dragging)
            return;
        root.radiusCommitted(root.dragRadius);
        root.dragging = false;
        root.snapping = false;
    }

    // ── Projection ──

    // a circle in perspective projects to a rotated conic, which none of the
    // arc paths can draw, so the rim goes out as a polyline through sampled
    // points. anything behind the camera projects to nonsense, so the points
    // in front are collected into the runs that can actually be stroked
    property var ringRuns: []

    property var handles: []

    function _projectRing() {
        if (!view3d || !view3d.camera) {
            ringRuns = [];
            return;
        }

        var projected = [];
        var firstBehind = -1;
        for (var i = 0; i < root.segments; i++) {
            var screen = view3d.mapFrom3DScene(root.parentWorld.times(root._rimPoint(i * 360 / root.segments, root.liveRadius)));
            projected.push(screen);
            if (firstBehind < 0 && screen.z <= 0)
                firstBehind = i;
        }

        // the whole rim is in front, so it draws as one loop, closed by
        // repeating the point it started on
        if (firstBehind < 0) {
            var loop = projected.map(point => Qt.point(point.x, point.y));
            loop.push(loop[0]);
            ringRuns = [loop];
            return;
        }

        // otherwise walk from a point already behind the camera, so that no
        // run has to wrap around the seam to stay in one piece
        var runs = [];
        var current = [];
        for (var step = 1; step <= root.segments; step++) {
            var point = projected[(firstBehind + step) % root.segments];
            if (point.z > 0) {
                current.push(Qt.point(point.x, point.y));
                continue;
            }
            if (current.length > 1)
                runs.push(current);
            current = [];
        }
        if (current.length > 1)
            runs.push(current);

        ringRuns = runs;
    }

    function _projectHandles() {
        if (!view3d || !view3d.camera) {
            handles = [];
            return;
        }

        var projected = [];
        for (var i = 0; i < root.handleAngles.length; i++) {
            var angle = root.handleAngles[i];
            var screen = view3d.mapFrom3DScene(root.parentWorld.times(root._rimPoint(angle, root.liveRadius)));
            projected.push({
                visible: screen.z > 0,
                angle: angle,
                x: screen.x,
                y: screen.y,
                depth: screen.z
            });
        }

        handles = projected;
    }

    property var _lastProbe: null

    // the rim at two opposite handles is enough to notice the circle moving,
    // turning, resizing or the camera going anywhere
    function _screenProbe() {
        if (!view3d || !view3d.camera)
            return null;

        var near = view3d.mapFrom3DScene(root.parentWorld.times(root._rimPoint(0, root.liveRadius)));
        var far = view3d.mapFrom3DScene(root.parentWorld.times(root._rimPoint(180, root.liveRadius)));
        return [near.x, near.y, near.z, far.x, far.y, far.z];
    }

    FrameAnimation {
        running: root.visible && root.ready

        onTriggered: {
            var probe = root._screenProbe();
            if (probe === null)
                return;

            var last = root._lastProbe;
            if (last !== null && probe.every((value, i) => Math.abs(value - last[i]) < 0.0001))
                return;

            root._projectRing();
            root._projectHandles();
            root._lastProbe = probe;
        }
    }

    // ── Hit testing ──

    property int hoverIndex: -1

    function _handleAt(x, y) {
        var bestIndex = -1;
        var bestDepth = Infinity;

        for (var i = 0; i < root.handles.length; i++) {
            var handle = root.handles[i];
            if (!handle.visible)
                continue;

            var hit = HitTester.testCenterHandleHit(Qt.point(x, y), Qt.point(handle.x, handle.y), root.hitRadius);
            if (hit.hit && handle.depth < bestDepth) {
                bestDepth = handle.depth;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    // ── Rendering ──

    Repeater {
        model: root.ringRuns

        delegate: Shape {
            id: runShape

            required property var modelData

            anchors.fill: parent
            visible: runShape.modelData.length > 1
            preferredRendererType: Shape.CurveRenderer

            ShapePath {
                strokeColor: root.ringColor
                strokeWidth: root.lineWidth
                fillColor: "transparent"

                PathPolyline {
                    path: runShape.modelData
                }
            }
        }
    }

    Repeater {
        model: root.handleAngles.length

        delegate: Shape {
            id: puckShape

            required property int index

            readonly property var handle: root.handles.length === root.handleAngles.length ? root.handles[puckShape.index] : null
            readonly property bool active: root.dragging ? root._dragAngle === root.handleAngles[puckShape.index] : root.hoverIndex === puckShape.index

            readonly property color tint: puckShape.active ? Qt.lighter(root.ringColor, 1.35) : root.ringColor
            readonly property real halfSize: root.puckSize / 2

            anchors.fill: parent
            visible: puckShape.handle !== null && puckShape.handle.visible
            preferredRendererType: Shape.CurveRenderer

            ShapePath {
                strokeColor: puckShape.tint
                strokeWidth: 1.5
                fillColor: puckShape.active ? puckShape.tint : "transparent"

                startX: (puckShape.handle ? puckShape.handle.x : 0) - puckShape.halfSize
                startY: (puckShape.handle ? puckShape.handle.y : 0) - puckShape.halfSize

                PathLine {
                    x: (puckShape.handle ? puckShape.handle.x : 0) + puckShape.halfSize
                    y: (puckShape.handle ? puckShape.handle.y : 0) - puckShape.halfSize
                }
                PathLine {
                    x: (puckShape.handle ? puckShape.handle.x : 0) + puckShape.halfSize
                    y: (puckShape.handle ? puckShape.handle.y : 0) + puckShape.halfSize
                }
                PathLine {
                    x: (puckShape.handle ? puckShape.handle.x : 0) - puckShape.halfSize
                    y: (puckShape.handle ? puckShape.handle.y : 0) + puckShape.halfSize
                }
                PathLine {
                    x: (puckShape.handle ? puckShape.handle.x : 0) - puckShape.halfSize
                    y: (puckShape.handle ? puckShape.handle.y : 0) - puckShape.halfSize
                }
            }
        }
    }

    // ── Interaction ──

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        enabled: root.ready
        preventStealing: root.dragging

        onPositionChanged: mouse => {
            if (root.dragging) {
                root._updateDrag(mouse.x, mouse.y, mouse.modifiers);
                mouse.accepted = true;
                return;
            }
            root.hoverIndex = root._handleAt(mouse.x, mouse.y);
            mouse.accepted = false;
        }

        onPressed: mouse => {
            var index = root._handleAt(mouse.x, mouse.y);
            if (index < 0) {
                mouse.accepted = false;
                return;
            }

            root._beginDrag(root.handleAngles[index], mouse.x, mouse.y);
            root.hoverIndex = index;
            mouse.accepted = true;
        }

        onReleased: mouse => {
            if (root.dragging) {
                root._endDrag();
                mouse.accepted = true;
            } else {
                mouse.accepted = false;
            }
        }

        onCanceled: {
            if (root.dragging)
                root._endDrag();
            root.hoverIndex = -1;
        }

        onExited: {
            if (!root.dragging)
                root.hoverIndex = -1;
        }
    }

    onReadyChanged: {
        if (!ready) {
            hoverIndex = -1;
            if (dragging)
                _endDrag();
        }
    }
}
