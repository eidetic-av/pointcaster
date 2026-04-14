import QtQuick
import QtQuick3D

CustomMaterial {
    shadingMode: CustomMaterial.Unshaded
    vertexShader: "qrc:/qt/qml/Pointcaster/Geometry/shaders/point_cloud.vert"
    fragmentShader: "qrc:/qt/qml/Pointcaster/Geometry/shaders/point_cloud.frag"
}