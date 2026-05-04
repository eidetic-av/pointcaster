import QtQuick
import QtQuick3D

CustomMaterial {
    vertexShader: "qrc:/qt/qml/Pointcaster/Geometry/shaders/point_cloud.vert"
    fragmentShader: "qrc:/qt/qml/Pointcaster/Geometry/shaders/point_cloud.frag"

    shadingMode: CustomMaterial.Unshaded
    cullMode: Material.NoCulling

    property real uPointSize: 1.0
}