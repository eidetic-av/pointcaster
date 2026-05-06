import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Pointcaster 1.0

RowLayout {
    id: root

    property var alignmentController: null

    signal backToPicking()
    signal applyRequested()

    spacing: Math.round(Scaling.uiScale * 12)

    Item {
        Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
    }

    IconButton {
        text: "Picking"
        tooltip: "Return to keypoint pair picking"
        iconSource: FontAwesome.icon("solid/arrow-left-long")
        onClicked: root.backToPicking()
    }

    Rectangle {
        width: 1
        Layout.fillHeight: true
        Layout.topMargin: Math.round(Scaling.uiScale * 8)
        Layout.bottomMargin: Math.round(Scaling.uiScale * 8)
        color: ThemeColors.middark
    }

    Column {
        spacing: Math.round(Scaling.uiScale * 2)
        Layout.preferredWidth: Math.round(Scaling.uiScale * 160)

        Label {
            text: "Correspondence: " + correspondenceSlider.value.toFixed(0)
            font: Scaling.uiSmallFont
            color: ThemeColors.midlight
        }

        Slider {
            id: correspondenceSlider
            width: parent.width
            from: 5
            to: 200
            stepSize: 1
            value: root.alignmentController ? root.alignmentController.maxCorrespondenceDistance : 50
            onMoved: {
                if (root.alignmentController)
                    root.alignmentController.maxCorrespondenceDistance = value
            }
        }
    }

    Column {
        spacing: Math.round(Scaling.uiScale * 2)
        Layout.preferredWidth: Math.round(Scaling.uiScale * 120)

        Label {
            text: "Voxel size: " + voxelSlider.value.toFixed(1)
            font: Scaling.uiSmallFont
            color: ThemeColors.midlight
        }

        Slider {
            id: voxelSlider
            width: parent.width
            from: 0
            to: 20
            stepSize: 0.5
            value: root.alignmentController ? root.alignmentController.voxelLeafSize : 5
            onMoved: {
                if (root.alignmentController)
                    root.alignmentController.voxelLeafSize = value
            }
        }
    }

    Column {
        spacing: Math.round(Scaling.uiScale * 2)
        Layout.preferredWidth: Math.round(Scaling.uiScale * 120)

        Label {
            text: "Iterations: " + iterationsSlider.value.toFixed(0)
            font: Scaling.uiSmallFont
            color: ThemeColors.midlight
        }

        Slider {
            id: iterationsSlider
            width: parent.width
            from: 10
            to: 200
            stepSize: 10
            value: root.alignmentController ? root.alignmentController.maxIterations : 50
            onMoved: {
                if (root.alignmentController)
                    root.alignmentController.maxIterations = value
            }
        }
    }

    Rectangle {
        width: 1
        Layout.fillHeight: true
        Layout.topMargin: Math.round(Scaling.uiScale * 8)
        Layout.bottomMargin: Math.round(Scaling.uiScale * 8)
        color: ThemeColors.middark
    }

    IconButton {
        text: "Refine"
        tooltip: "Run ICP refinement on the current alignment"
        enabled: root.alignmentController && root.alignmentController.hasResult && !root.alignmentController.refining
        onClicked: root.alignmentController.refine()
    }

    Label {
        visible: root.alignmentController && root.alignmentController.fitnessScore > 0
        text: "fitness: " + (root.alignmentController ? root.alignmentController.fitnessScore.toFixed(4) : "")
        font: Scaling.uiSmallFont
        color: ThemeColors.midlight
    }

    Item {
        Layout.fillWidth: true
    }

    IconButton {
        text: "Apply to Session"
        tooltip: "Apply alignment transform to session configuration"
        iconSource: FontAwesome.icon("solid/floppy-disk")
        onClicked: root.applyRequested()
    }

    Item {
        Layout.preferredWidth: Math.round(Scaling.uiScale * 12)
    }
}