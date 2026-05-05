pragma Singleton
import QtQuick

QtObject {
    id: registry

    property ListModel model: ListModel {}

    function indexOfKey(key) {
        for (let i = 0; i < model.count; ++i) {
            if (model.get(i).key === key)
                return i;
        }
        return -1;
    }

    function addPage(key, title, componentOrUrl) {
        const idx = indexOfKey(key);
        const row = {
            key: key,
            title: title,
            page: componentOrUrl
        };

        if (idx < 0)
            model.append(row);
        else
            model.set(idx, row);
    }

    function removePage(key) {
        const idx = indexOfKey(key);
        if (idx >= 0)
            model.remove(idx);
    }

    function clear() {
        model.clear();
    }
}
