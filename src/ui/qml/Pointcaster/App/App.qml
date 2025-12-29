import QtQuick
import QtQuick.Controls.Fusion
import QtQuick.Controls
import QtQuick.Layouts
import Pointcaster.Controls
import com.kdab.dockwidgets as KDDW

ApplicationWindow {
    id: root
    visible: true
    title: "Pointcaster"

    minimumWidth: 1200
    minimumHeight: 800

    palette {
        window:            "#1e1e2e"
        windowText:        "#cdd6f4"
        base:              "#181825"
        text:              "#cdd6f4"
        button:            "#313244"
        buttonText:        "#cdd6f4"
        highlight:         "#89b4fa"
        highlightedText:   "#1e1e2e"
    }

    FontLoader {
        id: regularFont
        source: "qrc:/fonts/AtkinsonHyperlegibleNext-Regular.otf"
    }

    font.family: regularFont.name

    menuBar : MenuBar {

        Menu {
            title: qsTr("&File")

            Action { text: qsTr("&Save layout") }
            Action { text: qsTr("&Restore layout") }

            MenuSeparator{}
            Action {
                text: qsTr("&Quit")
                onTriggered: Qt.quit()
            }
        }

        Menu {
            title: qsTr("&Window")
        }

        Menu {
            title: qsTr("&Help")

            Action {
                text: qsTr("&About")
                onTriggered: aboutPopup.open()
            }
        }
    }

    KDDW.DockingArea {
        id: dockingArea
        anchors.fill: parent
        uniqueName: "MainDockingArea"

        KDDW.DockWidget {
            id: browseDock
            uniqueName: "browseDock"
            title: "Browse"

            Rectangle {
                width: 100
                height: 100
                color: "#2E8BC0"
            }
        }

        KDDW.DockWidget {
            id: detailDock
            uniqueName: "detailDock"
            title: "Details"

            Rectangle {
                width: 200
                height: 700
                color: "#8B2EC0"
            }
        }

        Component.onCompleted: {
            addDockWidget(browseDock, KDDW.KDDockWidgets.Location_OnLeft)
            addDockWidget(detailDock,
                          KDDW.KDDockWidgets.Location_OnRight,
                          browseDock)
        }
    }

    Popup {
        id: aboutPopup
        modal: true
        dim: true
        focus: true
        width: 360
        height: 200

        x: (parent.width - width) / 2
        y: (parent.height - height) / 2

        background: Rectangle {
            color: "#181825"
            radius: 8
            border.color: "#89b4fa"
        }

        Column {
            anchors.centerIn: parent
            spacing: 12

            Label {
                text: "Pointcaster\n\nAbout dialog placeholder"
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
            }

            Button {
                text: "Close"
                onClicked: aboutPopup.close()
            }
        }
    }

}

