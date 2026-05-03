import QtQuick
import QtQuick3D

CustomCamera {
    id: root

    // 0 = fully perspective, 1 = fully orthographic
    property real blend: 0
    // Perspective params
    property real nearPlane: 1
    property real farPlane: 250000
    property real fovYRadians: 60 * Math.PI / 180
    // ortho params
    property real orthoHalfHeight: z * 0.6
    // OrbitCameraController expects these to exist
    property real clipNear
    property real clipFar

    function lerp(a, b, t) {
        return a + (b - a) * t;
    }

    function setBlend(targetValue, animate) {
        projectionBlendAnim.stop();
        if (!animate) {
            blend = targetValue;
            return;
        }
        projectionBlendAnim.from = blend;
        projectionBlendAnim.to = targetValue;
        // easing/duration differs by direction because the matrix blend is non-linear
        const ascending = projectionBlendAnim.to > projectionBlendAnim.from;
        projectionBlendAnim.easing.type = ascending ? Easing.OutExpo : Easing.InCubic;
        projectionBlendAnim.duration = ascending ? 150 : 350;
        projectionBlendAnim.start();
    }

    z: root.defaultCameraDistance
    projection: {
        const aspect = view.width > 0 ? (view.width / view.height) : 1;
        const t = blend;
        // --- Perspective ---
        const cot = Math.cos(fovYRadians * 0.5) / Math.sin(fovYRadians * 0.5);
        const p00 = cot / aspect;
        const p11 = cot;
        const p22 = -(nearPlane + farPlane) / (farPlane - nearPlane);
        const p23 = -(2 * nearPlane * farPlane) / (farPlane - nearPlane);
        const p32 = -1;
        const p33 = 0;
        // --- Orthographic ---
        const top = orthoHalfHeight;
        const bottom = -orthoHalfHeight;
        const right = orthoHalfHeight * aspect;
        const left = -orthoHalfHeight * aspect;
        const o00 = 2 / (right - left);
        const o11 = 2 / (top - bottom);
        // handles clipping dist, make it pretty much infinite for ortho
        const o22 = -1e-08;
        const o23 = 0;
        const o32 = 0;
        const o33 = 1;
        const m00 = lerp(p00, o00, t);
        const m11 = lerp(p11, o11, t);
        const m22 = lerp(p22, o22, t);
        const m23 = lerp(p23, o23, t);
        const m32 = lerp(p32, o32, t);
        const m33 = lerp(p33, o33, t);
        return Qt.matrix4x4(m00, 0, 0, 0, 0, m11, 0, 0, 0, 0, m22, m23, 0, 0, m32, m33);
    }

    NumberAnimation {
        // duration/easing set in setBlend()

        id: projectionBlendAnim

        target: root
        property: "blend"
    }
}
