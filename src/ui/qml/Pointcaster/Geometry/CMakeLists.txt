
qt_add_qml_module(PointcasterGeometry
	URI Pointcaster.Geometry VERSION 1.0
    DEPENDENCIES
        QtQuick3D
    SOURCES
        PointCloudGeometry.h
        PointCloudGeometry.cc
    RESOURCES
        shaders/point_cloud.vert
        shaders/point_cloud.frag
)

target_link_libraries(PointcasterGeometry PRIVATE
    pointcaster::core
    Qt6::Quick3D
)

target_include_directories(PointcasterGeometry
    PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/src")

install(DIRECTORY "${QT_QML_OUTPUT_DIRECTORY}/Pointcaster/Geometry"
	DESTINATION "${CMAKE_INSTALL_QMLDIR}/Pointcaster"
)

install(TARGETS PointcasterGeometry
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

target_precompile_headers(PointcasterGeometry PRIVATE
    <QByteArray>
    <QObject>
    <QQuick3DGeometry>
    <QtQmlIntegration/qqmlintegration.h>
    <QString>
    <QVariant>
    <QUrl>
    <QVector3D>
)