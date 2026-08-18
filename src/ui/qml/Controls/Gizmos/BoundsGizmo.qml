pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Shapes
import QtQuick3D

import Pointcaster 1.0

import Gizmo3D

// Push/pull editing of an AABB
Item {
    id: root

    property View3D view3d: null

    // committed corners in parent's space
    property vector3d minPosition: Qt.vector3d(0, 0, 0)
    property vector3d maxPosition: Qt.vector3d(0, 0, 0)

    // parent space -> world space.
    property matrix4x4 parentWorld: Qt.matrix4x4()

    property real limitLow: -1000
    property real limitHigh: 1000

    property real minExtent: 1

    // Held during a drag: mirror resizes about the box centre
    property int mirrorModifier: Qt.AltModifier

    property int snapModifier: Qt.ControlModifier

    // In whatever unit the parent space uses, so the owner sets it.
    property real snapIncrement: 10

    property color boxColor: ThemeColors.yellow
    property real lineWidth: 1

    // faces take the axis they move along; min/max are told apart by fill, so
    // the hue keeps meaning the axis
    property color xAxisColor: ThemeColors.red
    property color yAxisColor: ThemeColors.green
    property color zAxisColor: ThemeColors.blue

    property real stalkLength: 20
    property real puckSize: 10
    property real hitRadius: 13

    anchors.fill: parent

    // ── Live drag state ──

    property bool dragging: false
    property bool draggingMax: false

    // Whether the modifiers were down at the last mouse move.
    property bool mirroring: false
    property bool snapping: false

    property vector3d dragMin: Qt.vector3d(0, 0, 0)
    property vector3d dragMax: Qt.vector3d(0, 0, 0)

    readonly property vector3d liveMin: dragging ? dragMin : minPosition
    readonly property vector3d liveMax: dragging ? dragMax : maxPosition

    signal boundsPreview(vector3d newMin, vector3d newMax)
    signal boundsCommitted(vector3d newMin, vector3d newMax)

    readonly property bool ready: view3d !== null

    // ── Geometry helpers ──

    function _component(vector, axis) {
        return axis === 0 ? vector.x : axis === 1 ? vector.y : vector.z;
    }

    function _withComponent(vector, axis, value) {
        return Qt.vector3d(axis === 0 ? value : vector.x, axis === 1 ? value : vector.y, axis === 2 ? value : vector.z);
    }

    function _axisColor(axis) {
        return axis === 0 ? root.xAxisColor : axis === 1 ? root.yAxisColor : root.zAxisColor;
    }

    function _axisWorld(axis) {
        var matrix = root.parentWorld;
        var column = axis === 0 ? Qt.vector3d(matrix.m11, matrix.m21, matrix.m31) : axis === 1 ? Qt.vector3d(matrix.m12, matrix.m22, matrix.m32) : Qt.vector3d(matrix.m13, matrix.m23, matrix.m33);
        return GizmoMath.normalize(column);
    }

    function _faceCentre(axis, isMax) {
        var low = root.liveMin;
        var high = root.liveMax;
        var centre = Qt.vector3d((low.x + high.x) / 2, (low.y + high.y) / 2, (low.z + high.z) / 2);
        return _withComponent(centre, axis, _component(isMax ? high : low, axis));
    }

    // ── Drag plumbing ──

    property int _dragAxis: -1
    property vector3d _dragFaceWorld: Qt.vector3d(0, 0, 0)
    property vector3d _dragAxisWorld: Qt.vector3d(0, 0, 0)
    property real _dragInitialParameter: 0

    property vector3d _dragStartMin: Qt.vector3d(0, 0, 0)
    property vector3d _dragStartMax: Qt.vector3d(0, 0, 0)

    function _axisParameter(screenX, screenY) {
        var ray = GizmoMath.getCameraRay(root.view3d, Qt.point(screenX, screenY));
        return -GizmoMath.closestPointOnAxisToRay(ray.origin, ray.direction, root._dragFaceWorld, root._dragAxisWorld);
    }

    function _beginDrag(axis, isMax, screenX, screenY) {
        root._dragAxis = axis;
        root.draggingMax = isMax;
        root._dragStartMin = root.minPosition;
        root._dragStartMax = root.maxPosition;
        // seeded before dragging flips, since liveMin/liveMax read these once it does
        root.dragMin = root.minPosition;
        root.dragMax = root.maxPosition;
        root._dragFaceWorld = root.parentWorld.times(_faceCentre(axis, isMax));
        root._dragAxisWorld = _axisWorld(axis);
        root._dragInitialParameter = _axisParameter(screenX, screenY);
        root.dragging = true;
    }

    function _applyDrag(value) {
        var axis = root._dragAxis;
        var low = _component(root._dragStartMin, axis);
        var high = _component(root._dragStartMax, axis);
        var nextLow = low;
        var nextHigh = high;

        if (root.mirroring) {
            // the centre holds still and both faces move out together
            var centre = (low + high) / 2;
            var halfExtent = root.draggingMax ? value - centre : centre - value;
            var maxHalfExtent = Math.min(root.limitHigh - centre, centre - root.limitLow);
            halfExtent = Math.max(root.minExtent / 2, Math.min(halfExtent, maxHalfExtent));
            nextLow = centre - halfExtent;
            nextHigh = centre + halfExtent;
        } else if (root.draggingMax) {
            nextHigh = Math.min(root.limitHigh, Math.max(value, low + root.minExtent));
        } else {
            nextLow = Math.max(root.limitLow, Math.min(value, high - root.minExtent));
        }

        root.dragMin = _withComponent(root._dragStartMin, axis, nextLow);
        root.dragMax = _withComponent(root._dragStartMax, axis, nextHigh);
    }

    function _updateDrag(screenX, screenY, modifiers) {
        if (!root.dragging)
            return;

        root.mirroring = (modifiers & root.mirrorModifier) !== 0;
        root.snapping = (modifiers & root.snapModifier) !== 0;

        var delta = _axisParameter(screenX, screenY) - root._dragInitialParameter;
        var moved = Qt.vector3d(root._dragFaceWorld.x + root._dragAxisWorld.x * delta, root._dragFaceWorld.y + root._dragAxisWorld.y * delta, root._dragFaceWorld.z + root._dragAxisWorld.z * delta);

        // back to parent space...
        var localPoint = root.parentWorld.inverted().times(moved);
        var value = _component(localPoint, root._dragAxis);

        if (root.snapping) {
            value = GizmoMath.snapValueAbsolute(value, root.snapIncrement);
        }

        root._applyDrag(value);
        root.boundsPreview(root.dragMin, root.dragMax);
    }

    function _endDrag() {
        if (!root.dragging)
            return;
        root.boundsCommitted(root.dragMin, root.dragMax);
        root.dragging = false;
        root.mirroring = false;
        root.snapping = false;
        root._dragAxis = -1;
    }

    // ── Projection ──

    property var screenCorners: []

    property var faces: []

    // Corner ordering is a 3-bit index: bit 0 = x, bit 1 = y, bit 2 = z, where a
    // set bit takes the value from the max corner.
    readonly property var edgePairs: [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    function _projectCorners() {
        if (!view3d || !view3d.camera) {
            screenCorners = [];
            return;
        }

        var low = root.liveMin;
        var high = root.liveMax;
        var corners = [];

        for (var i = 0; i < 8; i++) {
            var localCorner = Qt.vector3d((i & 1) ? high.x : low.x, (i & 2) ? high.y : low.y, (i & 4) ? high.z : low.z);
            var worldCorner = root.parentWorld.times(localCorner);
            var screenCorner = view3d.mapFrom3DScene(worldCorner);
            corners.push({
                x: screenCorner.x,
                y: screenCorner.y,
                inFront: screenCorner.z > 0
            });
        }

        screenCorners = corners;
    }

    function _projectFaces() {
        if (!view3d || !view3d.camera) {
            faces = [];
            return;
        }

        var low = root.liveMin;
        var high = root.liveMax;

        // Probe distance for reading the outward direction in screen space.
        var probeDistance = Math.max(0.5, Math.max(high.x - low.x, Math.max(high.y - low.y, high.z - low.z)) * 0.02);
        var projectedFaces = [];

        for (var i = 0; i < 6; i++) {
            var axis = i >> 1;
            var isMax = (i & 1) === 1;

            var worldCentre = root.parentWorld.times(_faceCentre(axis, isMax));
            var screenCentre = view3d.mapFrom3DScene(worldCentre);

            var axisDirection = _axisWorld(axis);
            var normal = isMax ? axisDirection : Qt.vector3d(-axisDirection.x, -axisDirection.y, -axisDirection.z);

            // the inward point is always inside the box, so it stays in front of the near plane
            var inward = Qt.vector3d(worldCentre.x - normal.x * probeDistance, worldCentre.y - normal.y * probeDistance, worldCentre.z - normal.z * probeDistance);
            var inwardScreen = view3d.mapFrom3DScene(inward);

            var outwardX = screenCentre.x - inwardScreen.x;
            var outwardY = screenCentre.y - inwardScreen.y;
            var length = Math.sqrt(outwardX * outwardX + outwardY * outwardY);
            if (length < 0.0001) {
                outwardX = 0;
                outwardY = -1;
            } else {
                outwardX /= length;
                outwardY /= length;
            }

            projectedFaces.push({
                visible: screenCentre.z > 0,
                axis: axis,
                isMax: isMax,
                anchorX: screenCentre.x,
                anchorY: screenCentre.y,
                puckX: screenCentre.x + outwardX * root.stalkLength,
                puckY: screenCentre.y + outwardY * root.stalkLength,
                depth: screenCentre.z
            });
        }

        faces = projectedFaces;
    }

    property var _lastProbe: null

    // returns the box's two opposite corners
    function _screenProbe() {
        if (!view3d || !view3d.camera)
            return null;

        var low = view3d.mapFrom3DScene(root.parentWorld.times(root.liveMin));
        var high = view3d.mapFrom3DScene(root.parentWorld.times(root.liveMax));
        return [low.x, low.y, low.z, high.x, high.y, high.z];
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

            root._projectCorners();
            root._projectFaces();
            root._lastProbe = probe;
        }
    }

    // ── Hit testing ──

    property int hoverIndex: -1

    function _faceAt(x, y) {
        var bestIndex = -1;
        var bestDepth = Infinity;

        for (var i = 0; i < root.faces.length; i++) {
            var face = root.faces[i];
            if (!face.visible)
                continue;

            var hit = HitTester.testCenterHandleHit(Qt.point(x, y), Qt.point(face.puckX, face.puckY), root.hitRadius);
            if (hit.hit && face.depth < bestDepth) {
                bestDepth = face.depth;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    // ── Rendering ──

    // per edge
    Repeater {
        model: 12

        delegate: Shape {
            id: edgeShape

            required property int index

            readonly property var cornerPair: root.edgePairs[edgeShape.index]
            readonly property var startCorner: root.screenCorners.length === 8 ? root.screenCorners[edgeShape.cornerPair[0]] : null
            readonly property var endCorner: root.screenCorners.length === 8 ? root.screenCorners[edgeShape.cornerPair[1]] : null

            anchors.fill: parent
            visible: edgeShape.startCorner !== null && edgeShape.endCorner !== null && edgeShape.startCorner.inFront && edgeShape.endCorner.inFront
            preferredRendererType: Shape.CurveRenderer

            ShapePath {
                strokeColor: root.boxColor
                strokeWidth: root.lineWidth
                fillColor: "transparent"

                startX: edgeShape.startCorner ? edgeShape.startCorner.x : 0
                startY: edgeShape.startCorner ? edgeShape.startCorner.y : 0

                PathLine {
                    x: edgeShape.endCorner ? edgeShape.endCorner.x : 0
                    y: edgeShape.endCorner ? edgeShape.endCorner.y : 0
                }
            }
        }
    }

    Repeater {
        model: 6

        delegate: Shape {
            id: puckShape

            required property int index

            readonly property var face: root.faces.length === 6 ? root.faces[puckShape.index] : null
            // While mirroring, the opposite face lights up too
            readonly property bool active: root.dragging ? (root._dragAxis === (puckShape.index >> 1) && (root.mirroring || root.draggingMax === ((puckShape.index & 1) === 1))) : root.hoverIndex === puckShape.index

            readonly property bool shown: puckShape.face !== null && puckShape.face.visible

            readonly property color tint: puckShape.face ? (puckShape.active ? Qt.lighter(root._axisColor(puckShape.face.axis), 1.35) : root._axisColor(puckShape.face.axis)) : "transparent"
            readonly property real halfSize: root.puckSize / 2

            anchors.fill: parent
            visible: puckShape.shown
            preferredRendererType: Shape.CurveRenderer

            // stalk
            ShapePath {
                strokeColor: puckShape.tint
                strokeWidth: puckShape.active ? 2 : 1.5
                fillColor: "transparent"

                startX: puckShape.face ? puckShape.face.anchorX : 0
                startY: puckShape.face ? puckShape.face.anchorY : 0

                PathLine {
                    x: puckShape.face ? puckShape.face.puckX : 0
                    y: puckShape.face ? puckShape.face.puckY : 0
                }
            }

            // grabber "puck"
            ShapePath {
                strokeColor: puckShape.tint
                strokeWidth: 1.5
                fillColor: (puckShape.face && puckShape.face.isMax) ? puckShape.tint : "transparent"

                startX: (puckShape.face ? puckShape.face.puckX : 0) - puckShape.halfSize
                startY: (puckShape.face ? puckShape.face.puckY : 0) - puckShape.halfSize

                PathLine {
                    x: (puckShape.face ? puckShape.face.puckX : 0) + puckShape.halfSize
                    y: (puckShape.face ? puckShape.face.puckY : 0) - puckShape.halfSize
                }
                PathLine {
                    x: (puckShape.face ? puckShape.face.puckX : 0) + puckShape.halfSize
                    y: (puckShape.face ? puckShape.face.puckY : 0) + puckShape.halfSize
                }
                PathLine {
                    x: (puckShape.face ? puckShape.face.puckX : 0) - puckShape.halfSize
                    y: (puckShape.face ? puckShape.face.puckY : 0) + puckShape.halfSize
                }
                PathLine {
                    x: (puckShape.face ? puckShape.face.puckX : 0) - puckShape.halfSize
                    y: (puckShape.face ? puckShape.face.puckY : 0) - puckShape.halfSize
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
            root.hoverIndex = root._faceAt(mouse.x, mouse.y);
            mouse.accepted = false;
        }

        onPressed: mouse => {
            var index = root._faceAt(mouse.x, mouse.y);
            if (index < 0) {
                mouse.accepted = false;
                return;
            }

            root._beginDrag(index >> 1, (index & 1) === 1, mouse.x, mouse.y);
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
