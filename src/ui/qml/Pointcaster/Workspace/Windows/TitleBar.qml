import QtQuick 2.6
import "qrc:/kddockwidgets/qtquick/views/qml/" as KDDW

KDDW.TitleBarBase {
    id: root

    implicitHeight: 30
    // height: implicitHeight

    Rectangle {
        anchors.fill: parent
        color: DarkPalette.middark
    }

    // a single 'tab'
    Rectangle {
        id: singleTab
        height: parent.height
        width: Math.min(parent.width, 120)
        anchors.left: parent.left
        anchors.top: parent.top

        color: DarkPalette.mid

        Rectangle {
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: 1
            color: DarkPalette.highlight
        }

        Text {
            text: root.title
            anchors.verticalCenter: parent.verticalCenter
            anchors.centerIn: parent
            elide: Text.ElideRight
            font.pixelSize: 15
            color: DarkPalette.text
            opacity: 1
        }
    }
}
