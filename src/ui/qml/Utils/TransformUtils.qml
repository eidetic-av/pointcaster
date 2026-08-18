pragma Singleton

import QtQuick

// Rotation helpers for transform configs

QtObject {
    function quaternionFromAxisAngle(axis, angleDegrees) {
        const half = angleDegrees * (Math.PI / 180) * 0.5;
        const s = Math.sin(half);
        return Qt.quaternion(Math.cos(half), axis.x * s, axis.y * s, axis.z * s);
    }

    // config euler (degrees) -> orientation. Matches Qt's fromEulerAngles.
    function quaternionFromEuler(euler) {
        if (!euler || euler.x === undefined)
            return Qt.quaternion(1, 0, 0, 0);
        const qx = quaternionFromAxisAngle(Qt.vector3d(1, 0, 0), euler.x);
        const qy = quaternionFromAxisAngle(Qt.vector3d(0, 1, 0), euler.y);
        const qz = quaternionFromAxisAngle(Qt.vector3d(0, 0, 1), euler.z);
        return qy.times(qx).times(qz);
    }

    function eulerFromQuaternion(q) {
        return q.normalized().toEulerAngles();
    }

    function positionFromMatrix(m) {
        return Qt.vector3d(m.m14, m.m24, m.m34);
    }

    function scaleFromMatrix(m) {
        return Qt.vector3d(_columnLength(m.m11, m.m21, m.m31), _columnLength(m.m12, m.m22, m.m32), _columnLength(m.m13, m.m23, m.m33));
    }

    function _columnLength(x, y, z) {
        const len = Math.sqrt(x * x + y * y + z * z);
        return len < 1e-9 ? 1 : len;
    }

    function rotationFromMatrix(m) {
        const cx = _normalizedOrDefault(Qt.vector3d(m.m11, m.m21, m.m31), Qt.vector3d(1, 0, 0));
        const cy = _normalizedOrDefault(Qt.vector3d(m.m12, m.m22, m.m32), Qt.vector3d(0, 1, 0));
        const cz = _normalizedOrDefault(Qt.vector3d(m.m13, m.m23, m.m33), Qt.vector3d(0, 0, 1));
        return _quaternionFromBasis(cx, cy, cz);
    }

    function _normalizedOrDefault(v, fallback) {
        const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (len < 1e-9)
            return fallback;
        return Qt.vector3d(v.x / len, v.y / len, v.z / len);
    }

    function _quaternionFromBasis(cx, cy, cz) {
        const m00 = cx.x, m10 = cx.y, m20 = cx.z;
        const m01 = cy.x, m11 = cy.y, m21 = cy.z;
        const m02 = cz.x, m12 = cz.y, m22 = cz.z;

        const trace = m00 + m11 + m22;
        if (trace > 0) {
            const s = Math.sqrt(trace + 1) * 2;
            return Qt.quaternion(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s);
        }
        if (m00 > m11 && m00 > m22) {
            const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
            return Qt.quaternion((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s);
        }
        if (m11 > m22) {
            const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
            return Qt.quaternion((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s);
        }
        const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
        return Qt.quaternion((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s);
    }
}
