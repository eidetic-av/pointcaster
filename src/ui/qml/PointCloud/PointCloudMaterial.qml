import QtQuick
import QtQuick3D

CustomMaterial {
    vertexShader: "qrc:/qt/qml/Pointcaster/PointCloud/Shaders/PointCloud.vert"
    fragmentShader: "qrc:/qt/qml/Pointcaster/PointCloud/Shaders/PointCloud.frag"

    shadingMode: CustomMaterial.Unshaded
    cullMode: Material.NoCulling

    property real uPointSize: 0.5  // world units (cm)

    property real uViewportHeight: 1080.0

    property real uMinPointPixels: 1.0
    property real uMaxPointPixels: 64.0
}
