pragma Singleton
import QtQuick

QtObject {
    function icon(faPath, color) {
        let hex;
        if (!color) hex = DarkPalette.text.toString().slice(1);
        else hex = color.toString().slice(1);
        return "image://fa/" + faPath + "?color=" + hex;
    }
}