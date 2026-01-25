pragma Singleton

import QtQuick

import Pointcaster 1.0

QtObject {
    id: root

    property real uiScale: AppSettings.uiScale

    property real basePointSize: 11.5

    readonly property real pointSize: basePointSize * uiScale
    readonly property real smallPointSize: basePointSize * 0.80 * uiScale
    readonly property real headerPointSize: basePointSize * 1.10 * uiScale

    readonly property font uiFont: Qt.font({ pointSize: pointSize })
    readonly property font uiSmallFont: Qt.font({ pointSize: smallPointSize, weight: Font.Bold })
    readonly property font uiHeaderFont: Qt.font({ pointSize: headerPointSize, weight: Font.Medium })
}
