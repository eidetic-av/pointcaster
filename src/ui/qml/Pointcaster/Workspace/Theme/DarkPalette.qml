pragma Singleton

import QtQuick
import QtQuick.Controls

QtObject {

    property Palette palette: Palette {
        accent: "#80CBC4"
        window: "#0F111A"
        windowText: "#E5E9F0"

        base: "#0F111A"
        alternateBase: "#00010A"

        text: "#E5E9F0"
        placeholderText: "#4C566A"

        button: "#3B4252"
        buttonText: "#E5E9F0"

        highlight: "#80CBC4"
        highlightedText: "#00010A"

        light: "#E5E9F0"
        midlight: "#4C566A"  
        mid:      "#3B4252"
        dark:     "#151823"
        shadow:   "#00010A"
    }

    property color red: "#ff4151"
    property color green: "#aecf7f"
    property color blue: "#81a1c1"

    property color success: green
    property color error: red
    property color neutralSuccess: blue 
    property color inactive : palette.mid

    // Aliases for access
    property color accent: palette.accent
    property color window: palette.window
    property color windowText: palette.windowText
    property color base: palette.base
    property color alternateBase: palette.alternateBase
    property color text: palette.text
    property color placeholderText: palette.placeholderText
    property color button: palette.button
    property color buttonText: palette.buttonText
    property color highlight: palette.highlight
    property color highlightedText: palette.highlightedText
    property color light: palette.light
    property color midlight: palette.midlight
    property color mid: palette.mid
    property color middark: "#2A2F3A"
    property color almostdark: "#1E2230" 
    property color dark: palette.dark
    property color shadow: palette.shadow
}
