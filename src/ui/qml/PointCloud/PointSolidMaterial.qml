import QtQuick
import QtQuick3D

CustomMaterial {
    vertexShader: "qrc:/qt/qml/Pointcaster/PointCloud/Shaders/PointCloud.vert"
    fragmentShader: "qrc:/qt/qml/Pointcaster/PointCloud/Shaders/PointSolid.frag"

    shadingMode: CustomMaterial.Unshaded
    cullMode: Material.NoCulling

    property real uPointSize: 1.0
    property color uColor: "white"
}
