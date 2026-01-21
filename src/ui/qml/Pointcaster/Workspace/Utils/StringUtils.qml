pragma Singleton
import QtQuick

QtObject {
    function titleFromSnake(str) {
        if (!str)
            return "";

        return str
            .split("_")
            .filter(function(s) { return s.length > 0; })
            .map(function(s) {
                return s.charAt(0).toUpperCase() + s.slice(1);
            })
            .join(" ");
    }
}
