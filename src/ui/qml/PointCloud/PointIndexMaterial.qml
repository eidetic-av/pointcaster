import QtQuick
import QtQuick3D

CustomMaterial {
    vertexShader: "qrc:/qt/qml/Pointcaster/PointCloud/Shaders/PointIndex.vert"
    fragmentShader: "qrc:/qt/qml/Pointcaster/PointCloud/Shaders/PointCloud.frag"

    shadingMode: CustomMaterial.Unshaded
    cullMode: Material.NoCulling

    property real uPointSize: 1.0
}