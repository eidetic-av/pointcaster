import QtQuick 2.6

import "qrc:/kddockwidgets/qtquick/views/qml/" as KDDW

KDDW.TitleBarBase {
    id: root

    Rectangle {
        anchors.fill: parent
        color: "#303340"
    }

    Text {
        text: root.title
        color: "#f0f0f0"
        anchors {
            left: parent.left
            leftMargin: 10
            verticalCenter: parent.verticalCenter
        }
    }
}
