import QtQuick
import com.kdab.dockwidgets as KDDW

import Pointcaster.Workspace 1.0

// Item {
//     anchors.fill: parent

    KDDW.DockingArea {
        id: dockingArea
        anchors.fill: parent
        uniqueName: "SessionDockingArea"

        affinities: ["view"]

        KDDW.DockWidget {
            id: session2
            uniqueName: "session2"
            title: "session_2"
            affinities: ["view"]
            SessionView {}
        }

        KDDW.DockWidget {
            id: session3
            uniqueName: "session3"
            title: "session_3"
            affinities: ["view"]
            SessionView {}
        }

        Component.onCompleted: {
            addDockWidget(session2, KDDW.KDDockWidgets.Location_OnLeft);
            addDockWidget(session3, KDDW.KDDockWidgets.Location_OnRight);
        }
    }
// }