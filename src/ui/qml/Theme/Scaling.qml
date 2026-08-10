pragma Singleton

import QtQuick

import Pointcaster 1.0

QtObject {
    id: root

    property real uiScale: AppSettings.uiScale

    property real basePointSize: 10.5

    readonly property real pointSize: basePointSize * uiScale
    readonly property real smallPointSize: basePointSize * 0.80 * uiScale
    readonly property real headerPointSize: basePointSize * 1.10 * uiScale

    readonly property font uiFont: Qt.font({
        pointSize: pointSize
    })
    readonly property font uiSmallFont: Qt.font({
        pointSize: smallPointSize,
        weight: Font.Bold,
        letterSpacing: 1.5
    })
    readonly property font headerSmallFont: Qt.font({
        pointSize: basePointSize * 0.90 * uiScale,
        weight: Font.Normal,
        letterSpacing: 0.9
    })
    readonly property font fieldLabelFont: Qt.font({
        pointSize: 10 * uiScale
    })
    readonly property font uiHeaderFont: Qt.font({
        pointSize: headerPointSize,
        weight: Font.Medium
    })

    readonly property real monoFontSize: basePointSize * 0.90 * uiScale
    readonly property font monoFont: Qt.font({
        pointSize: monoFontSize,
        family: "Atkinson Hyperlegible Mono",
        weight: Font.Medium,
        letterSpacing: -0.4
    })
}
