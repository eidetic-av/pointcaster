import QtQuick
import QtQuick.Layouts

import Pointcaster 1.0

SettingsPage {
    id: root

    title: "Performance"

    SettingsSection {
        title: "Threads"

        SettingsRow {
            label: "Worker threads"
            description: "Used for device processing and sequence loading."

            DragInt {
                font: Scaling.uiFont
                Layout.preferredWidth: Math.round(110 * Scaling.uiScale)
                minValue: 1
                maxValue: 256
                defaultValue: AppSettings.defaultWorkerThreads
                boundValue: AppSettings.workerThreads
                onCommitValue: function (v) {
                    AppSettings.workerThreads = v;
                }
            }
        }

        SettingsRow {
            label: "Recorder file write threads"

            DragInt {
                font: Scaling.uiFont
                Layout.preferredWidth: Math.round(110 * Scaling.uiScale)
                minValue: 1
                maxValue: 64
                defaultValue: AppSettings.defaultFileWriterThreads
                boundValue: AppSettings.fileWriterThreads
                onCommitValue: function (v) {
                    AppSettings.fileWriterThreads = v;
                }
            }
        }
    }
}
