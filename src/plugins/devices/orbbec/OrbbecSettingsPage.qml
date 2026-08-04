import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

SettingsPage {
    id: root

    title: "Orbbec"

    readonly property string sdkVersionKey: "devices/orbbec/sdkVersion"

    SettingsSection {
        SettingsRow {
            label: "SDK version"
            description: "Requires restart."

            ComboBox {
                id: sdkVersionCombo

                font: Scaling.uiFont
                textRole: "text"
                valueRole: "value"
                Layout.preferredWidth: Math.round(180 * Scaling.uiScale)

                model: [
                    {
                        text: "Orbbec SDK v2",
                        value: 2
                    },
                    {
                        text: "Orbbec SDK v1",
                        value: 1
                    }
                ]

                Component.onCompleted: {
                    const saved = Number(AppSettings.value(root.sdkVersionKey, 2));
                    const idx = indexOfValue(saved === 1 ? 1 : 2);
                    if (idx >= 0)
                        currentIndex = idx;
                }

                onActivated: {
                    AppSettings.setValue(root.sdkVersionKey, currentValue);
                    AppSettings.sync();
                }
            }
        }
    }
}
