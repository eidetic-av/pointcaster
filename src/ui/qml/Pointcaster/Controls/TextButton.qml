import QtQuick
import QtQuick.Controls

Control {
    id: root
    property alias text: label.text

    implicitWidth: 120
    implicitHeight: 40

    background: Rectangle {
        radius: 8
        color: "#ffc0cb"  // light pink for now
    }

    contentItem: Text {
        id: label
        anchors.centerIn: parent
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: 14
        color: "black"
    }
}
